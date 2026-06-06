from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def recall_at_k(sim: np.ndarray, k: int) -> float:
    # sim: [N,N], row i is image i vs all texts, correct match is i
    ranks = np.argsort(-sim, axis=1)
    hits = 0
    for i in range(sim.shape[0]):
        if i in ranks[i, :k]:
            hits += 1
    return hits / sim.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--text_encoder", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--qwen_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    WSI_REPORT_ROOT = os.path.dirname(os.path.abspath(__file__))
    WSI_REPORT_ROOT = os.path.dirname(WSI_REPORT_ROOT)  # .../WSI_Report
    if WSI_REPORT_ROOT not in sys.path:
        sys.path.insert(0, WSI_REPORT_ROOT)

    from src.runtime_paths import add_repo_root_to_sys_path

    add_repo_root_to_sys_path()

    from src.dataset import ROIFeatureDataset, collate_pairs
    from src.modeling.coca_model import CoCaConfig, WSIReportCoCa
    from src.modeling.text_encoder import HFTextEncoder
    from src.text_processing import to_structured_report

    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("Please install transformers (see requirements.extra.txt)") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Text encoder
    txt_tok = AutoTokenizer.from_pretrained(args.text_encoder, use_fast=True)
    txt_backbone = AutoModel.from_pretrained(args.text_encoder)
    txt_enc = HFTextEncoder(txt_backbone, hidden_size=txt_backbone.config.hidden_size, proj_dim=512).to(device)

    ds = ROIFeatureDataset(args.csv, args.feature_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pairs)

    # Need Qwen only to infer llm_dim for model construction.
    qwen = AutoModelForCausalLM.from_pretrained(args.qwen_model, torch_dtype=torch.float16)
    llm_dim = qwen.get_input_embeddings().embedding_dim
    del qwen

    coca_cfg = CoCaConfig()
    model = WSIReportCoCa(llm_dim=llm_dim, cfg=coca_cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    txt_enc.load_state_dict(ckpt["text_encoder"], strict=True)

    model.eval()
    txt_enc.eval()

    z_imgs: List[torch.Tensor] = []
    z_txts: List[torch.Tensor] = []

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        for batch in dl:
            roi_feat = batch["roi_feat"].to(device)
            roi_mask = batch["roi_mask"].to(device)
            roi_mag = batch.get("roi_mag")
            if roi_mag is not None:
                roi_mag = roi_mag.to(device)
            reports = [to_structured_report(r) for r in batch["report_text"]]

            mags_list = batch.get("mags")
            if mags_list is None and roi_mag is not None:
                mags_list = [int(x.item()) for x in torch.unique(roi_mag[roi_mag >= 0])]

            out = model(roi_feat, roi_mask=roi_mask, roi_mag=roi_mag, mags=mags_list)
            z_imgs.append(out["z_img"].float().cpu())

            txt = txt_tok(reports, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            z_txt = txt_enc(txt.input_ids, txt.attention_mask)
            z_txts.append(z_txt.float().cpu())

    z_img = torch.cat(z_imgs, dim=0).numpy()
    z_txt = torch.cat(z_txts, dim=0).numpy()

    sim = z_img @ z_txt.T

    print("N=", sim.shape[0])
    for k in [1, 5, 10, 20]:
        print(f"R@{k}: {recall_at_k(sim, k):.4f}")


if __name__ == "__main__":
    main()
