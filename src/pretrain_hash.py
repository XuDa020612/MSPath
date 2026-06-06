import argparse
import logging
import math
import os
import sys
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Make relative imports work
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from dataset import ROIFeatureDataset, collate_pairs
from modeling.losses import clip_contrastive_loss_learnable
from modeling.text_encoder import HFTextEncoder
from modeling.wsi_encoder import DeepHashSelector
from utils import setup_seed
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_init() -> Tuple[int, int, int, torch.device]:
    if not is_distributed():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device

def log0(msg: str):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)

class DeepHashPretrainer(nn.Module):
    def __init__(self, tile_dim, hash_bits, hash_rank, hash_guide_alpha, hash_chunk_size, hash_on_mag, proj_dim):
        super().__init__()
        self.hash_bits = hash_bits
        self.hash_on_mag = hash_on_mag if isinstance(hash_on_mag, list) else [hash_on_mag]
        
        self.selector = DeepHashSelector(
            dim=tile_dim,
            num_patches=2048, # Placeholder, dynamic inside
            hash_bits=hash_bits,
            guide_alpha=hash_guide_alpha,
            chunk_size=hash_chunk_size,
            hash_rank=hash_rank,
        )
        self.visual_proj = nn.Linear(hash_bits, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def _chunked_hash_proj(self, x_b: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.selector.chunk_size)
        if chunk_size <= 0 or x_b.shape[0] <= chunk_size:
            return torch.tanh(self.selector.hash_proj2(self.selector.hash_proj1(x_b)))

        chunks = []
        for start in range(0, x_b.shape[0], chunk_size):
            end = min(start + chunk_size, x_b.shape[0])
            chunk = x_b[start:end]
            chunks.append(torch.tanh(self.selector.hash_proj2(self.selector.hash_proj1(chunk))))
        return torch.cat(chunks, dim=0)

    def forward(self, features_by_mag: Dict[int, torch.Tensor], masks_by_mag: Dict[int, torch.Tensor]):
        wsi_hashes = []
        for mag in self.hash_on_mag:
            if mag not in features_by_mag:
                continue
                
            x = features_by_mag[mag]      #[B, N, D]
            mask = masks_by_mag[mag]      #[B, N]
            bsz, n, d = x.shape
            
            mag_wsi_hashes = []
            for b in range(bsz):
                valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    mag_wsi_hashes.append(torch.zeros(self.hash_bits, device=x.device, dtype=x.dtype))
                    continue
                
                x_b = x[b, valid_idx] # [Nv, D]
                # Forward chunked to save peak memory if needed
                h_b = self._chunked_hash_proj(x_b) # [Nv, HashBits]
                g_b = self.selector.guide_proj(x_b).squeeze(-1) # [Nv]
                
                weights = torch.softmax(g_b, dim=0)
                
                # Attentional pool: important patches contribute more to WSI hash
                wsi_h = torch.sum(weights.unsqueeze(-1) * h_b, dim=0) # [HashBits]
                mag_wsi_hashes.append(wsi_h)
            
            mag_wsi_hashes = torch.stack(mag_wsi_hashes, dim=0) # [B, HashBits]
            wsi_hashes.append(mag_wsi_hashes)
            
        if not wsi_hashes:
            first_val = next(iter(features_by_mag.values()))
            bsz = first_val.shape[0]
            wsi_hash = torch.zeros(bsz, self.hash_bits, device=first_val.device, dtype=first_val.dtype)
        else:
            wsi_hash = torch.mean(torch.stack(wsi_hashes, dim=0), dim=0) # [B, HashBits]
            
        z_img = F.normalize(self.visual_proj(wsi_hash), dim=-1)
        return z_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    rank, world_size, local_rank, device = ddp_init()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["Data"]
    model_cfg = cfg["Model"]
    text_cfg = cfg["Text"]
    train_cfg = cfg["Train"]

    setup_seed(train_cfg.get("seed", 42) + rank)
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log0(f"Starting Hash Pretraining -> {args.output_dir}")

    # Dataset
    train_ds = ROIFeatureDataset(
        csv_path=data_cfg["csv_path"],
        feature_dir=data_cfg["feature_dir"],
        report_col=data_cfg.get("report_col", "report_text"),
        target_mags=data_cfg.get("mags", [0, 1]),
    )
    sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    bsz = int(train_cfg["batch_size"])
    dl = DataLoader(
        train_ds, batch_size=bsz, sampler=sampler, shuffle=(sampler is None),
        num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=True,
        collate_fn=collate_pairs, drop_last=True
    )

    # Models
    txt_backbone = AutoModel.from_pretrained(text_cfg["text_encoder_name"])
    txt_hidden = getattr(txt_backbone.config, "hidden_size", None)
    tokenizer = AutoTokenizer.from_pretrained(text_cfg["text_encoder_name"])
    
    txt_enc = HFTextEncoder(txt_backbone, hidden_size=txt_hidden, proj_dim=int(text_cfg["proj_dim"])).to(device)
    if not bool(text_cfg.get("finetune_backbone", False)):
        for p in txt_enc.backbone.parameters():
            p.requires_grad = False
        
    model = DeepHashPretrainer(
        tile_dim=model_cfg.get("tile_dim", 768),
        hash_bits=model_cfg.get("hash_bits", 64),
        hash_rank=model_cfg.get("hash_rank", 64),
        hash_guide_alpha=model_cfg.get("hash_guide_alpha", 0.2),
        hash_chunk_size=model_cfg.get("hash_chunk_size", 1024),
        hash_on_mag=model_cfg.get("hash_on_mag", [0, 1]),
        proj_dim=int(text_cfg["proj_dim"])
    ).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank])
        txt_enc = DDP(txt_enc, device_ids=[local_rank], find_unused_parameters=True)

    params = list(model.parameters()) + list(txt_enc.parameters())
    optim = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01))
    )
    
    max_steps = int(train_cfg["steps"])
    num_warmup_steps = int(max_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps)
    
    use_amp = bool(train_cfg.get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    txt_enc.train()
    
    step = 0
    best_loss = float('inf')
    dl_it = iter(dl)
    optim.zero_grad(set_to_none=True)
    
    while step < max_steps:
        if sampler is not None: sampler.set_epoch(step)
        try:
            batch = next(dl_it)
        except StopIteration:
            dl_it = iter(dl)
            batch = next(dl_it)
            
        mags = batch["mags"]
        features_by_mag = {m: batch[f"feat_{m}"].to(device, non_blocking=True) for m in mags}
        masks_by_mag = {m: batch[f"mask_{m}"].to(device, non_blocking=True) for m in mags}
        
        texts = batch["report_text"]
        toks = tokenizer(
            texts,
            return_tensors="pt",
            max_length=int(text_cfg.get("max_length", 512)),
            padding="max_length",
            truncation=True,
        )
        input_ids = toks["input_ids"].to(device, non_blocking=True)
        attention_mask = toks["attention_mask"].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            z_img = model(features_by_mag, masks_by_mag)
            z_txt = txt_enc(input_ids, attention_mask)
            mod = model.module if hasattr(model, "module") else model
            loss = clip_contrastive_loss_learnable(z_img, z_txt, mod.logit_scale)
        
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        scheduler.step()
        optim.zero_grad(set_to_none=True)
        
        if rank == 0:
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                save_path = os.path.join(args.output_dir, "hash_pretrained_weight_best.pt")
                mod = model.module if hasattr(model, "module") else model
                txt_state_dict = txt_enc.module.state_dict() if hasattr(txt_enc, "module") else txt_enc.state_dict()
                save_dict = {
                    "hash_model": mod.state_dict(),
                    "text_encoder": txt_state_dict,
                    "config": cfg,
                }
                torch.save(save_dict, save_path)
            
            if step % args.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                log0(f"step={step} lr={current_lr:.2e} loss={current_loss:.4f} (best: {best_loss:.4f})")
            
        step += 1
        
    if rank == 0:
        log0(f"Training completed. Best Hash weights saved with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
