from __future__ import annotations

import argparse
import logging
import os
import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Make relative imports work when invoked from repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from dataset import ROIFeatureDataset, collate_pairs
from modeling.coca_model import CoCaConfig, WSIReportCoCa
from modeling.losses import clip_contrastive_loss, clip_contrastive_loss_learnable
from src.text_processing import to_structured_report
from utils import setup_seed


def build_structured_prompt() -> str:
    return (
        "Summarize the key pathological findings based on the provided whole-slide images and clinical background when available.\n"
        "Write one or more English prose paragraphs without bullet points or section headers.\n"
    )


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_init() -> Tuple[int, int, int, torch.device]:
    """Initialize DDP if launched with torchrun."""
    if not is_distributed():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl" if os.name != "nt" else "gloo"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank, device


def ddp_all_gather(t: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks (no grad needed)."""
    if not is_distributed():
        return t
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return torch.cat(out, dim=0)


def save_checkpoint(path: str, step: int, model: torch.nn.Module, txt_enc: torch.nn.Module, optim: torch.optim.Optimizer, scheduler: Any = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "text_encoder": txt_enc.state_dict(),
        "optimizer": optim.state_dict(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, txt_enc: torch.nn.Module, optim: torch.optim.Optimizer, scheduler: Any = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model_keys = model.load_state_dict(ckpt["model"], strict=False)
    txt_keys = txt_enc.load_state_dict(ckpt["text_encoder"], strict=False)
    if getattr(model_keys, "missing_keys", None) or getattr(model_keys, "unexpected_keys", None):
        print(f"[ckpt] model missing={len(model_keys.missing_keys)} unexpected={len(model_keys.unexpected_keys)}")
    if getattr(txt_keys, "missing_keys", None) or getattr(txt_keys, "unexpected_keys", None):
        print(f"[ckpt] text_encoder missing={len(txt_keys.missing_keys)} unexpected={len(txt_keys.unexpected_keys)}")
    optim.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))


def read_cfg(path: str) -> Dict[str, Any]:
    # Reuse CHIEF util if available.
    try:
        from utils.utils import read_yaml

        return dict(read_yaml(path))
    except Exception:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


@torch.no_grad()
def run_validation(val_dl, model, txt_enc, qwen, qwen_tok, txt_tok, instruction_embeds, prompt_len, device, train_cfg, instruction_text):
    model.eval()
    txt_enc.eval()
    
    total_loss = 0.0
    steps = 0
    valid_batches = 0
    
    lambda_contrast = float(train_cfg["lambda_contrast"])
    lambda_gen = float(train_cfg["lambda_gen"])
    lambda_recon = float(train_cfg.get("lambda_recon", 1.0))
    
    for batch in val_dl:
        # We no longer skip small batches completely. Contrastive loss will just be 0 for batch_size=1.
        
        mags_list = batch["mags"]
        features_by_mag = {}
        masks_by_mag = {}
        coords_by_mag = {}
        
        for mag in mags_list:
            features_by_mag[mag] = batch[f"feat_{mag}"].to(device, non_blocking=True)
            masks_by_mag[mag] = batch[f"mask_{mag}"].to(device, non_blocking=True)
            if batch.get(f"coords_{mag}") is not None:
                coords_by_mag[mag] = batch[f"coords_{mag}"].to(device, non_blocking=True)
        
        level0_size = batch.get("level0_size", None)
        if level0_size is not None:
            level0_size = level0_size.to(device, non_blocking=True)
            
        reports_raw = batch["report_text"]
        reports = [to_structured_report(r) for r in reports_raw]
        clinical_history = batch.get("clinical_history", [""] * len(reports))
        
        bsz = len(reports)
        inst_embed = instruction_embeds.expand(bsz, -1, -1)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
             out = model(
                features_by_mag=features_by_mag,
                masks_by_mag=masks_by_mag,
                coords_by_mag=coords_by_mag if coords_by_mag else None,
                level0_size=level0_size,
                instruction_tokens=inst_embed,
            )
             z_img = out["z_img"]
             prefix = out["prefix"]

             txt = txt_tok(
                reports,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
             ).to(device)
             z_txt = txt_enc(txt.input_ids, txt.attention_mask)

             m = model.module if hasattr(model, "module") else model
             loss_contrast = clip_contrastive_loss_learnable(z_img, z_txt, m.logit_scale)
             
             # Generation
             prompt_texts = []
             full_text = []
             for clin, r in zip(clinical_history, reports):
                 if clin and clin != "Clinical information not available.":
                     prompt = instruction_text + f"\n[Clinical History]: {clin}\n[Pathology Report]:\n"
                 else:
                     prompt = instruction_text + "\n[Pathology Report]:\n"
                 prompt_texts.append(prompt)
                 full_text.append(prompt + r + qwen_tok.eos_token)
             gen = qwen_tok(
                 full_text,
                 padding=True,
                 truncation=True,
                 max_length=512,
                 return_tensors="pt",
             ).to(device)
             tok_emb = qwen.get_input_embeddings()(gen.input_ids)
             inputs_embeds = torch.cat([prefix.to(tok_emb.dtype), tok_emb], dim=1)

             attn_mask = torch.cat(
                [torch.ones(prefix.shape[:2], device=device, dtype=gen.attention_mask.dtype), gen.attention_mask],
                dim=1,
             )

             # Labels: mask prefix tokens and the REAL prompt length of each sample
             labels_text = gen.input_ids.clone()
             prompt_tok = qwen_tok(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
             ).to(device)
             prompt_lens = prompt_tok.attention_mask.sum(dim=1)
             for bi in range(labels_text.size(0)):
                 pl = int(prompt_lens[bi].item())
                 labels_text[bi, :pl] = -100
             labels = torch.cat(
                [
                    torch.full(prefix.shape[:2], -100, device=device, dtype=labels_text.dtype),
                    labels_text,
                ],
                dim=1,
             )

             gen_out = qwen(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
             loss_gen = gen_out.loss
             
             loss_recon = out["loss_recon"]

             loss = (
                 lambda_contrast * loss_contrast
                 + lambda_gen * loss_gen
                 + lambda_recon * loss_recon
             )
             
             if not torch.isnan(loss) and not torch.isinf(loss):
                 total_loss += loss.item()
                 valid_batches += 1
        
        steps += 1
    
    model.train()
    txt_enc.train()
    
    if valid_batches == 0:
        return float('inf')
        
    return total_loss / valid_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PKG_ROOT, "configs", "sample.yaml"))
    parser.add_argument("--output_dir", default=os.path.join(PKG_ROOT, "outputs"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--log_file",
        default=None,
        help="Optional path to a log file. Default: <output_dir>/train.log (rank0 only)",
    )
    parser.add_argument("--instruction", default=None, help="Override instruction prompt text")
    parser.add_argument(
        "--llm_tune",
        choices=["auto", "frozen", "lora"],
        default="auto",
        help="Override LLM tuning strategy (default: config)",
    )
    args = parser.parse_args()

    rank, world_size, _, device = ddp_init()
    setup_seed(args.seed + rank)

    # Logging (rank0 only writes to file).
    logger: logging.Logger | None = None
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = args.log_file or os.path.join(args.output_dir, "train.log")
        logger = logging.getLogger("wsi_report_train")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False

    def log0(msg: str) -> None:
        if rank != 0:
            return
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    cfg = read_cfg(args.config)
    data_cfg = cfg["Data"]
    text_cfg = cfg["Text"]
    llm_cfg = cfg["LLM"]
    model_cfg = cfg["Model"]
    train_cfg = cfg["Train"]

    if rank == 0:
        log0(f"config={args.config}")
        log0(f"output_dir={args.output_dir}")
        log0(f"device={device.type} world_size={world_size}")
        log0(f"seed={args.seed}")

    # Lazy imports to keep scaffold usable without transformers installed.
    try:
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("Please install transformers (see requirements.extra.txt)") from e

    # Text encoder for retrieval alignment.
    txt_tok = AutoTokenizer.from_pretrained(text_cfg["text_encoder_name"], use_fast=False)
    if txt_tok.pad_token is None:
        txt_tok.pad_token = txt_tok.eos_token
    txt_backbone = AutoModel.from_pretrained(text_cfg["text_encoder_name"])
    txt_hidden = getattr(txt_backbone.config, "hidden_size", None)
    if txt_hidden is None:
        raise RuntimeError("Cannot infer text encoder hidden_size")

    from modeling.text_encoder import HFTextEncoder

    txt_enc = HFTextEncoder(txt_backbone, hidden_size=txt_hidden, proj_dim=int(text_cfg["proj_dim"]))
    txt_enc = txt_enc.to(device)

    # Text side: freeze backbone; train only pooling query + projection.
    for p in txt_enc.backbone.parameters():
        p.requires_grad = False
    for p in txt_enc.proj.parameters():
        p.requires_grad = True
    txt_enc.backbone.eval()

    # Qwen LLM (optionally tunable)
    qwen_tok = AutoTokenizer.from_pretrained(llm_cfg["qwen_model_name"], use_fast=True)
    if qwen_tok.pad_token is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        llm_cfg["qwen_model_name"], 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",  # Enable offloading
        offload_state_dict=True
    )
    # qwen = qwen.to(device)  # Handled by device_map
    
    # Enable gradient checkpointing for memory efficiency
    qwen.gradient_checkpointing_enable()

    llm_strategy = args.llm_tune if args.llm_tune != "auto" else llm_cfg.get("tune_strategy", "frozen")
    llm_strategy = llm_strategy.lower()
    trainable_llm_params: List[torch.nn.Parameter] = []
    if llm_strategy == "frozen":
        qwen.eval()
        for p in qwen.parameters():
            p.requires_grad = False
    elif llm_strategy == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise RuntimeError("Requested LoRA tuning but peft is not installed") from e

        lora_cfg = LoraConfig(
            r=int(llm_cfg.get("lora_r", 16)),
            lora_alpha=int(llm_cfg.get("lora_alpha", 32)),
            lora_dropout=float(llm_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=llm_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        qwen = get_peft_model(qwen, lora_cfg)
        qwen.train()
        trainable_llm_params = [p for p in qwen.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Unknown LLM tuning strategy: {llm_strategy}")

    llm_dim = qwen.get_input_embeddings().embedding_dim

    coca_cfg = CoCaConfig(
        tile_dim=int(model_cfg["tile_dim"]),
        roi_dim=int(model_cfg["roi_fused_dim"]),
        proj_dim=int(text_cfg["proj_dim"]),
        mil_hidden=int(model_cfg["mil_hidden"]),
        topk=int(model_cfg["topk"]),
        prefix_len=int(model_cfg["prefix_len"]),
        encoder_dropout=float(model_cfg.get("dropout", 0.25)),
        bottleneck_num_latents=int(model_cfg.get("bottleneck_num_latents", 64)),
        active_n_rois=int(model_cfg.get("active_n_rois", 2048)),
        hash_bits=int(model_cfg.get("hash_bits", 64)),
        hash_rank=int(model_cfg.get("hash_rank", 64)),
        hash_guide_alpha=float(model_cfg.get("hash_guide_alpha", 0.2)),
        hash_on_mag=model_cfg.get("hash_on_mag", 0),
        hash_chunk_size=int(model_cfg.get("hash_chunk_size", 1024)),
    )
    model = WSIReportCoCa(llm_dim=llm_dim, cfg=coca_cfg).to(device)

    # Load pretrained weights if specified
    pretrained_path = model_cfg.get("pretrained_path")
    if pretrained_path:
        if rank == 0:
            log0(f"Loading pretrained weights from {pretrained_path}")
        # Map to CPU first to avoid OOM or device mismatch
        ckpt = torch.load(pretrained_path, map_location="cpu")

        # Load Model
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        new_state_dict = {}
        model_state_dict = model.state_dict()
        for k, v in state_dict.items():
            # Strip 'module.' prefix if it exists (from DDP save)
            mapped_k = k[7:] if k.startswith("module.") else k
            
            # Skip if shapes do not match (e.g., when switching from 7B to 32B LLM)
            if mapped_k in model_state_dict:
                if list(v.shape) != list(model_state_dict[mapped_k].shape):
                    if rank == 0:
                        print(f"Skipping {mapped_k} due to shape mismatch: ckpt {list(v.shape)} != model {list(model_state_dict[mapped_k].shape)}")
                    continue
                    
            new_state_dict[mapped_k] = v
        
        m_keys = model.load_state_dict(new_state_dict, strict=False)
        if rank == 0:
            log0(f"[Pretrain] Model load: missing={len(m_keys.missing_keys)} unexpected={len(m_keys.unexpected_keys)}")

        # Load Text Encoder (optional but good for alignment)
        if "text_encoder" in ckpt:
            txt_state = ckpt["text_encoder"]
            new_txt_state = {}
            for k, v in txt_state.items():
                if k.startswith("module."):
                    new_txt_state[k[7:]] = v
                else:
                    new_txt_state[k] = v
            t_keys = txt_enc.load_state_dict(new_txt_state, strict=False)
            if rank == 0:
                log0(f"[Pretrain] TextEncoder load: missing={len(t_keys.missing_keys)} unexpected={len(t_keys.unexpected_keys)}")

    if rank == 0:
        log0(
            f"bottleneck_num_latents={coca_cfg.bottleneck_num_latents} "
            f"active_n_rois={coca_cfg.active_n_rois} hash_bits={coca_cfg.hash_bits} "
            f"hash_guide_alpha={coca_cfg.hash_guide_alpha} hash_on_mag={coca_cfg.hash_on_mag}"
        )

    # DDP wrap (only trainable modules)
    if is_distributed():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)
        txt_enc = DDP(txt_enc, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

    ds = ROIFeatureDataset(
        data_cfg["csv_path"], 
        data_cfg["feature_dir"], 
        report_col=data_cfg.get("report_col", "report_text"),
        n_rois=int(data_cfg.get("n_rois", -1)),
        n_rois_by_mag=data_cfg.get("n_rois_by_mag", None),
        target_mags=data_cfg.get("mags", [5, 10, 20])
    )
    sampler = DistributedSampler(ds, shuffle=True) if is_distributed() else None
    dl = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pairs,
        drop_last=True,
    )

    # Initialize Validation DataLoader
    val_dl = None
    
    if "val_csv" in data_cfg:
        val_csv = data_cfg["val_csv"]
    else:
        # Auto-infer val.csv name based on csv_path
        if "train.csv" in data_cfg["csv_path"]:
            val_csv = data_cfg["csv_path"].replace("train.csv", "val.csv")
        else:
            split_dir = os.path.dirname(data_cfg["csv_path"])
            val_csv = os.path.join(split_dir, "val.csv")
            
    if os.path.exists(val_csv):
        if rank == 0:
            log0(f"Loading validation set from {val_csv}")
        val_ds = ROIFeatureDataset(
            val_csv, 
            data_cfg["feature_dir"], 
            report_col=data_cfg.get("report_col", "report_text"),
            n_rois=int(data_cfg.get("n_rois", -1)),
            n_rois_by_mag=data_cfg.get("n_rois_by_mag", None),
            target_mags=data_cfg.get("mags", [5, 10, 20])
        )
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
        val_dl = DataLoader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]), # Same batch size as train
            shuffle=False,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_pairs,
            drop_last=False,
        )

    optim_params: List[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad] + [
        p for p in txt_enc.parameters() if p.requires_grad
    ]
    if trainable_llm_params:
        optim_params += trainable_llm_params

    optim = torch.optim.AdamW(
        optim_params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    
    # LR Scheduler (Linear Warmup + Cosine Decay)
    try:
        from transformers import get_cosine_schedule_with_warmup
    except ImportError:
        raise RuntimeError("Please install transformers.")

    max_steps = int(train_cfg["steps"])
    num_warmup_steps = int(max_steps * 0.1)  # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optim, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=max_steps
    )

    start_step = 0
    if args.resume is not None:
        # Only rank0 reads from disk, then broadcast.
        if rank == 0:
            start_step = load_checkpoint(args.resume, model.module if hasattr(model, "module") else model, txt_enc.module if hasattr(txt_enc, "module") else txt_enc, optim, scheduler)
        if is_distributed():
            step_t = torch.tensor([start_step], device=device, dtype=torch.int64)
            dist.broadcast(step_t, src=0)
            start_step = int(step_t.item())

    instruction_text = args.instruction or train_cfg.get("instruction") or build_structured_prompt()
    instruction_inputs = qwen_tok(instruction_text, add_special_tokens=True, return_tensors="pt")
    instruction_inputs = {k: v.to(device) for k, v in instruction_inputs.items()}
    instruction_embeds = qwen.get_input_embeddings()(instruction_inputs["input_ids"])
    prompt_len = int(instruction_inputs["input_ids"].shape[1])

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model.train()
    txt_enc.train()
    (txt_enc.module if hasattr(txt_enc, "module") else txt_enc).backbone.eval()

    max_steps = int(train_cfg["steps"])
    lambda_contrast = float(train_cfg["lambda_contrast"])
    lambda_gen = float(train_cfg["lambda_gen"])
    lambda_recon = float(train_cfg.get("lambda_recon", 5.0))

    step = start_step
    best_val_loss = float('inf')
    best_from_step = int(train_cfg.get("best_from_step", 0))
    best_to_step_cfg = train_cfg.get("best_to_step", None)
    best_to_step = int(best_to_step_cfg) if best_to_step_cfg is not None else max_steps
    stop_at_best_window_end = bool(train_cfg.get("stop_at_best_window_end", False))

    if best_to_step < best_from_step:
        raise ValueError(f"Invalid best window: best_from_step={best_from_step}, best_to_step={best_to_step}")

    if rank == 0:
        log0(
            f"best_window=[{best_from_step}, {best_to_step}] "
            f"stop_at_best_window_end={stop_at_best_window_end}"
        )
    
    dl_it = iter(dl)
    optim.zero_grad(set_to_none=True)
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(step)

        try:
            batch = next(dl_it)
        except StopIteration:
            dl_it = iter(dl)
            batch = next(dl_it)

        mags_list = batch["mags"]
        features_by_mag = {}
        masks_by_mag = {}
        coords_by_mag = {}
        
        for mag in mags_list:
            features_by_mag[mag] = batch[f"feat_{mag}"].to(device, non_blocking=True)
            masks_by_mag[mag] = batch[f"mask_{mag}"].to(device, non_blocking=True)
            if batch.get(f"coords_{mag}") is not None:
                coords_by_mag[mag] = batch[f"coords_{mag}"].to(device, non_blocking=True)
        
        level0_size = batch.get("level0_size", None)
        if level0_size is not None:
            level0_size = level0_size.to(device, non_blocking=True)
            
        reports_raw = batch["report_text"]
        reports = [to_structured_report(r) for r in reports_raw]
        clinical_history = batch.get("clinical_history", [""] * len(reports))
        
        bsz = len(reports)
        inst_embed = instruction_embeds.expand(bsz, -1, -1)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(
                features_by_mag=features_by_mag,
                masks_by_mag=masks_by_mag,
                coords_by_mag=coords_by_mag if coords_by_mag else None,
                level0_size=level0_size,
                instruction_tokens=inst_embed,
            )
            z_img = out["z_img"]
            prefix = out["prefix"]  # [B,P,Dl]

            # Text side embeddings for contrastive alignment.
            txt = txt_tok(
                reports,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            z_txt = txt_enc(txt.input_ids, txt.attention_mask)

            if is_distributed():
                z_img_all = ddp_all_gather(z_img.detach())
                z_txt_all = ddp_all_gather(z_txt.detach())
                # Compute loss locally using gathered embeddings (no gradient through other ranks).
                m = model.module if hasattr(model, "module") else model
                loss_contrast = clip_contrastive_loss_learnable(z_img, z_txt, m.logit_scale)  # local batch
                # Add a global-batch contrastive signal (no grad across ranks, but stabilizes).
                loss_contrast = 0.5 * loss_contrast + 0.5 * clip_contrastive_loss_learnable(z_img_all, z_txt_all, m.logit_scale)
            else:
                m = model.module if hasattr(model, "module") else model
                loss_contrast = clip_contrastive_loss_learnable(z_img, z_txt, m.logit_scale)

            # Generation: prefix + instruction + report
            # Inject clinical history into instruction with configurable probability.
            # Lower keep prob can reduce shortcut reliance and style drift.
            clinical_keep_prob = float(train_cfg.get("clinical_keep_prob", 0.5))
            prompt_texts = []
            full_text = []
            for clin, r in zip(clinical_history, reports):
                if clin and clin != "Clinical information not available." and random.random() < clinical_keep_prob:
                    prompt = instruction_text + f"\n[Clinical History]: {clin}\n[Pathology Report]:\n"
                else:
                    prompt = instruction_text + "\n[Pathology Report]:\n"

                prompt_texts.append(prompt)
                full_text.append(prompt + r + qwen_tok.eos_token)
            gen = qwen_tok(
                full_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            tok_emb = qwen.get_input_embeddings()(gen.input_ids)
            inputs_embeds = torch.cat([prefix.to(tok_emb.dtype), tok_emb], dim=1)

            attn_mask = torch.cat(
                [torch.ones(prefix.shape[:2], device=device, dtype=gen.attention_mask.dtype), gen.attention_mask],
                dim=1,
            )

            # Labels: ignore prefix tokens and REAL prompt tokens per sample.
            labels_text = gen.input_ids.clone()
            prompt_tok = qwen_tok(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            prompt_lens = prompt_tok.attention_mask.sum(dim=1)
            for bi in range(labels_text.size(0)):
                pl = int(prompt_lens[bi].item())
                labels_text[bi, :pl] = -100
            labels = torch.cat(
                [
                    torch.full(prefix.shape[:2], -100, device=device, dtype=labels_text.dtype),
                    labels_text,
                ],
                dim=1,
            )

            gen_out = qwen(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
            loss_gen = gen_out.loss
            
            loss_recon = out["loss_recon"]
            
            # Scale reconstruction loss is now handled by config "lambda_recon"
            loss = (
                lambda_contrast * loss_contrast
                + lambda_gen * loss_gen
                + lambda_recon * loss_recon
            ) / float(args.grad_accum)
             
            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.step(optim)
            scaler.update()
            scheduler.step()  # Update LR
            optim.zero_grad(set_to_none=True)

        if rank == 0 and step % int(args.log_every) == 0:
            current_lr = scheduler.get_last_lr()[0]
            m = model.module if hasattr(model, "module") else model
            temp = float((1.0 / m.logit_scale.exp()).detach().cpu().item())
            log0(
                f"step={step} lr={current_lr:.2e} loss={(loss.item()*args.grad_accum):.4f} "
                f"contrast={loss_contrast.item():.4f} gen={loss_gen.item():.4f} recon={loss_recon.item():.4f} "
                f"temp={temp:.4f}"
            )
        if step > 0:
            
            # Validation more frequently (every 100)
            if step % 100 == 0 and val_dl is not None:
                val_loss = run_validation(
                    val_dl, 
                    model, 
                    txt_enc, 
                    qwen, 
                    qwen_tok, 
                    txt_tok, 
                    instruction_embeds, 
                    prompt_len, 
                    device, 
                    train_cfg, 
                    instruction_text
                )
                
                # Sync val_loss across ranks to log consistent value (optional but good)
                if is_distributed():
                    val_loss_t = torch.tensor([val_loss], device=device)
                    dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_t.item() / dist.get_world_size()

                if rank == 0:
                    in_best_window = (step >= best_from_step) and (step <= best_to_step)
                    log0(
                        f"step={step} val_loss={val_loss:.4f} best_val={best_val_loss:.4f} "
                        f"in_best_window={in_best_window}"
                    )
                    if in_best_window and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt_path = os.path.join(args.output_dir, "checkpoint_best.pt")
                        log0(f"New best model found! Saving to {best_ckpt_path}")
                        save_checkpoint(
                            best_ckpt_path,
                            step,
                            model.module if hasattr(model, "module") else model,
                            txt_enc.module if hasattr(txt_enc, "module") else txt_enc,
                            optim,
                            scheduler
                        )

        if stop_at_best_window_end and step >= best_to_step:
            if rank == 0:
                log0(f"Reached best window end at step={step}. Early stopping.")
            step += 1
            break

        step += 1

    if rank == 0:
        ckpt_path = os.path.join(args.output_dir, "checkpoint_last.pt")
        save_checkpoint(
            ckpt_path,
            step,
            model.module if hasattr(model, "module") else model,
            txt_enc.module if hasattr(txt_enc, "module") else txt_enc,
            optim,
            scheduler
        )

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
