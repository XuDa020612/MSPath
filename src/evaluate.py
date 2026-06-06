import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import TreebankWordTokenizer
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

def compute_metrics(predictions, references):
    # BLEU-1 and BLEU-4
    # Switch to TreebankWordTokenizer to avoid dependency on punkt_tab (which is missing)
    tokenizer = TreebankWordTokenizer()
    
    # NLTK expects references as list of list of tokens: [[['ref1_word1', ...]], [['ref2_word1', ...]]]
    # Predictions as list of list of tokens: [['pred1_word1', ...], ['pred2_word1', ...]]
    
    ref_tokens = [[tokenizer.tokenize(r)] for r in references]
    pred_tokens = [tokenizer.tokenize(p) for p in predictions]
    
    # BLEU-1: weight (1, 0, 0, 0)
    bleu1 = nltk_corpus_bleu(ref_tokens, pred_tokens, weights=(1.0, 0.0, 0.0, 0.0))
    
    # BLEU-4: standard weights
    bleu4 = nltk_corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    
    # METEOR
    # NLTK meteor expects list of references (list of list of strings) and hypothesis (list of strings)
    # But actually, meteor_score takes single sentence ref (list of words) and single sent hyp (list of words)
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        # Tokenize for METEOR
        p_tok = tokenizer.tokenize(pred)
        r_tok = tokenizer.tokenize(ref)
        meteor_scores.append(meteor_score([r_tok], p_tok))
    avg_meteor = np.mean(meteor_scores)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)
            
    avg_rouge = {k: np.mean(v) for k, v in rouge_scores.items()}
    
    # Output in 0-1 scale to match TITAN
    return {
        'METEOR': avg_meteor,       # 0-1 scale
        'ROUGE-1': avg_rouge['rouge1'], # 0-1 scale
        'BLEU-1': bleu1,            # nltk returns 0-1
        'BLEU-4': bleu4             # nltk returns 0-1
    }

def clean_report(text: str, prompt: str = "") -> str:
    # 截断生成的文本，去掉它里面重复的 prompt 部分 (如果在开头)
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
        
    # 如果模型在生成合法预测之后又莫名其妙复读指令，从复读的地方截断
    if "请根据提供的全切片" in text:
        text = text.split("请根据提供的全切片")[0]
        
    # 如果模型复读了"Human:"这种格式，也截断
    if "Human:" in text:
        text = text.split("Human:")[0]

    # 可能模型还会再补入一个【病理报告】：的头，确保其头部被剥离（防御性保留剩余文本）
    if "【病理报告】：\n" in text:
        text = text.split("【病理报告】：\n")[-1]
    elif "【病理报告】：" in text:
        text = text.split("【病理报告】：")[-1]
        
    return text.strip()

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PKG_ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--test_csv", default=None, help="Path to the test csv file to evaluate on")
    parser.add_argument("--output_file", default="evaluation_results.csv", help="Where to save predictions")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--min_new_tokens", type=int, default=150)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=2)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Decoding: max_new_tokens={args.max_new_tokens}, "
        f"min_new_tokens={args.min_new_tokens}, num_beams={args.num_beams}, "
        f"length_penalty={args.length_penalty}"
    )

    # Load Config
    cfg = read_yaml_local(args.config)
    
    # Initialize Qwen (Frozen)
    llm_cfg = cfg["LLM"]
    print("Loading LLM...")
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

    # Initialize CoCa Model
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
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    keys = model.load_state_dict(new_state_dict, strict=False)
    if getattr(keys, "missing_keys", None) or getattr(keys, "unexpected_keys", None):
        print(f"[ckpt] model missing={len(keys.missing_keys)} unexpected={len(keys.unexpected_keys)}")
    model.eval()

    # Dataset
    data_cfg = cfg["Data"]
    
    if args.test_csv:
        test_csv = args.test_csv
    else:
        split_dir = os.path.dirname(data_cfg["csv_path"])
        test_csv = os.path.join(split_dir, "test.csv")
        if not os.path.exists(test_csv):
            print("Test Set not found, using Validation Set")
            test_csv = os.path.join(split_dir, "val.csv")
            
    print(f"Loading data from {test_csv}...")
    ds = ROIFeatureDataset(
        test_csv, 
        data_cfg["feature_dir"], 
        report_col=data_cfg.get("report_col", "report_text"),
        n_rois=8192,  # Limit ROIs for inference to prevent OOM on extremely large slides
        target_mags=data_cfg.get("mags", [5, 10, 20])
    )

    dl = DataLoader(ds, batch_size=1, collate_fn=collate_pairs, shuffle=False)
    
    # Instruction
    if "Train" in cfg and "instruction" in cfg["Train"]:
        instruction_text = cfg["Train"]["instruction"]
    else:
        instruction_text = (
            "Summarize the key pathological findings based on the provided whole-slide images.\n"
            "Write one or more English prose paragraphs without bullet points or section headers.\n"
        )
    inst_tokens = qwen_tok(instruction_text, return_tensors="pt").to(device)
    inst_embeds = qwen.get_input_embeddings()(inst_tokens.input_ids)

    predictions = []
    references = []
    case_ids = []

    print("Starting evaluation...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            # Skip cases with error messages in GT
            gt_report = batch["report_text"][0]
            if "Error:" in gt_report or "Timed out" in gt_report:
                continue

            case_id = batch["case_id"][0]
            clinical_history = batch.get("clinical_history", [""])[0]
            
            if clinical_history and clinical_history != "Clinical information not available.":
                prompt_text = instruction_text + f"\n[Clinical History]: {clinical_history}\n[Pathology Report]:\n"
            else:
                prompt_text = instruction_text + "\n[Pathology Report]:\n"
                
            inst_tokens = qwen_tok(prompt_text, return_tensors="pt").to(device)
            inst_embeds = qwen.get_input_embeddings()(inst_tokens.input_ids)
            
            # Prepare inputs
            mags_list = batch["mags"]
            features_by_mag = {}
            masks_by_mag = {}
            coords_by_mag = {}
            
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
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=qwen_tok.pad_token_id,
                    eos_token_id=qwen_tok.eos_token_id
                )
                
                generated_text = qwen_tok.batch_decode(gen_out, skip_special_tokens=True)[0]
                
            predictions.append(clean_report(generated_text, prompt_text))
            references.append(clean_report(gt_report))
            case_ids.append(case_id)

    # Save results
    df = pd.DataFrame({
        "case_id": case_ids,
        "ground_truth": references,
        "prediction": predictions
    })
    output_path = os.path.join(PKG_ROOT, "outputs", args.output_file)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # Compute Metrics
    metrics = compute_metrics(predictions, references)
    print("\nEvaluation Metrics:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
