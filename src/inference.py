import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

# Add src to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.dataset import ROIFeatureDataset, collate_pairs
from src.modeling.coca_model import CoCaConfig, WSIReportCoCa

def read_yaml_local(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_report(text):
    return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--input_csv", required=True, help="CSV with at least case_id")
    parser.add_argument("--feature_dir", required=True, help="Directory to the extracted .pt features")
    parser.add_argument("--output_file", default="inference_results.csv", help="Output CSV filename")
    parser.add_argument("--max_new_tokens", type=int, default=350)
    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config and Models
    cfg = read_yaml_local(args.config)
    llm_cfg = cfg["LLM"]
    qwen_tok = AutoTokenizer.from_pretrained(llm_cfg["qwen_model_name"], use_fast=False)
    if qwen_tok.pad_token is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    
    qwen = AutoModelForCausalLM.from_pretrained(
        llm_cfg["qwen_model_name"], 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    qwen.eval()
    llm_dim = qwen.get_input_embeddings().embedding_dim

    model_cfg = cfg["Model"]
    text_cfg = cfg["Text"]
    coca_cfg = CoCaConfig(
        tile_dim=int(model_cfg["tile_dim"]),
        roi_dim=int(model_cfg["roi_fused_dim"]),
        proj_dim=int(text_cfg["proj_dim"]),
        mil_hidden=int(model_cfg["mil_hidden"]),
        topk=int(model_cfg["topk"]),
        prefix_len=int(model_cfg["prefix_len"]),
    )
    model = WSIReportCoCa(llm_dim=llm_dim, cfg=coca_cfg).to(device)

    # 2. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. Handle Dataset (Bypass Ground Truth Requirement)
    data_cfg = cfg["Data"]
    df = pd.read_csv(args.input_csv)
    # Inject a dummy column so ROIFeatureDataset doesn't crash looking for labels
    df["dummy_report"] = "missing"
    temp_csv_path = os.path.join(PKG_ROOT, "data", "temp_inference.csv")
    df.to_csv(temp_csv_path, index=False)

    ds = ROIFeatureDataset(
        temp_csv_path, 
        args.feature_dir, 
        report_col="dummy_report",
        n_rois=8192,
        target_mags=data_cfg.get("mags", [0, 1, 2])
    )
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_pairs, shuffle=False)

    instruction_text = cfg.get("Train", {}).get("instruction", "Summarize the key pathological findings based on the provided whole-slide images.")

    # 4. Inference Loop
    predictions = []
    case_ids = []

    print(f"Starting inference on {len(ds)} samples...")
    with torch.no_grad():
        for batch in tqdm(dl):
            case_id = batch["case_id"][0]
            
            prompt_text = instruction_text + "\nReport:\n"
            inst_tokens = qwen_tok(prompt_text, return_tensors="pt").to(device)
            inst_embeds = qwen.get_input_embeddings()(inst_tokens.input_ids)

            mags_list = batch["mags"]
            features_by_mag, masks_by_mag, coords_by_mag = {}, {}, {}
            
            for mag in mags_list:
                features_by_mag[mag] = batch[f"feat_{mag}"].to(device)
                masks_by_mag[mag] = batch[f"mask_{mag}"].to(device)
                if batch.get(f"coords_{mag}") is not None:
                    coords_by_mag[mag] = batch[f"coords_{mag}"].to(device)
            
            level0_size = batch.get("level0_size")
            if level0_size is not None: 
                level0_size = level0_size.to(device)

            amp_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(
                    features_by_mag=features_by_mag,
                    masks_by_mag=masks_by_mag,
                    coords_by_mag=coords_by_mag if coords_by_mag else None,
                    level0_size=level0_size,
                    instruction_tokens=inst_embeds,
                )
                prefix = out["prefix"]
                inputs_embeds = torch.cat([prefix.to(inst_embeds.dtype), inst_embeds], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
                
                gen_out = qwen.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    pad_token_id=qwen_tok.pad_token_id,
                    eos_token_id=qwen_tok.eos_token_id
                )
                generated_text = qwen_tok.batch_decode(gen_out, skip_special_tokens=True)[0]
                
            predictions.append(clean_report(generated_text))
            case_ids.append(case_id)

    # 5. Clean up and Save
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    df_out = pd.DataFrame({
        "case_id": case_ids,
        "prediction": predictions
    })
    
    out_dir = os.path.join(PKG_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_file)
    df_out.to_csv(out_path, index=False)
    
    print(f"\n✅ Inference complete! Saved {len(df_out)} reports to {out_path}")

if __name__ == "__main__":
    main()
