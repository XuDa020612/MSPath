from __future__ import annotations

import argparse
import os
import sys
import math
import yaml
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# Make relative imports work when invoked from repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.modeling.wsi_encoder import DeepHashSelector
try:
    from src.utils import setup_seed
except ImportError:
    try:
        from utils import setup_seed
    except ImportError:
        # Fallback if utils cannot be imported (should not happen in correct env)
        def setup_seed(seed):
            import random
            import numpy as np
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

class OptimizedDeepHashSelector(DeepHashSelector):
    """
    Optimized version of DeepHashSelector for offline processing.
    Keeps all computations on GPU for maximum speed using vectorized FPS.
    """
    def __init__(
        self,
        dim: int,
        num_patches: int = 2048,
        hash_bits: int = 64,
        guide_alpha: float = 0.2,
        chunk_size: int = 1024,
        hash_rank: int = 64,
        candidate_mul: int = 4,
        min_norm_percentile: float = 0.2,
        norm_weight: float = 0.4,
        hash_weight: float = 0.4,
        guide_weight: float = 0.2,
        spatial_weight: float = 0.3,
        center_hash: bool = True,
        use_binary_hash: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            hash_bits=hash_bits,
            guide_alpha=guide_alpha,
            chunk_size=chunk_size,
            hash_rank=hash_rank,
        )
        self.candidate_mul = max(int(candidate_mul), 1)
        self.min_norm_percentile = float(min_norm_percentile)
        self.norm_weight = float(norm_weight)
        self.hash_weight = float(hash_weight)
        self.guide_weight = float(guide_weight)
        self.spatial_weight = float(spatial_weight)
        self.center_hash = bool(center_hash)
        self.use_binary_hash = bool(use_binary_hash)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        level0_size: Optional[Tuple[int, int]] = None,
    ):
        # x: [B, N, D] - In offline mode, usually B=1
        if self.num_patches <= 0 or x.shape[1] <= self.num_patches:
            return x, mask, coords

        bsz, n, d = x.shape
        k = self.num_patches

        # Pre-allocate outputs on GPU
        out_x = x.new_zeros((bsz, k, d))
        out_mask = torch.zeros((bsz, k), device=x.device, dtype=torch.bool) if mask is not None else None
        out_coords = coords.new_zeros((bsz, k, 2)) if coords is not None else None

        # Process each item in batch (usually just 1 in offline script)
        for b in range(bsz):
            if mask is not None:
                valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            else:
                valid_idx = torch.arange(n, device=x.device)

            nv = valid_idx.numel()
            if nv == 0:
                continue

            kb = min(k, nv)
            x_b = x[b, valid_idx]  # Use features directly on GPU
            
            # 1. Low-rank hash projection (All patches, chunked if needed)
            # Keep on GPU for speed; nv * hash_bits is typically small enough.
            h_chunks = []
            g_chunks = []
            n_chunks = []
            chunk = self.chunk_size
            for start in range(0, nv, chunk):
                end = min(start + chunk, nv)
                x_chunk = x_b[start:end]
                h_chunk = torch.tanh(self.hash_proj2(self.hash_proj1(x_chunk)))
                g_chunk = self.guide_proj(x_chunk).squeeze(-1)
                n_chunk = torch.norm(x_chunk, p=2, dim=-1)
                h_chunks.append(h_chunk)
                g_chunks.append(g_chunk)
                n_chunks.append(n_chunk)

            h_b = torch.cat(h_chunks, dim=0)
            guide_b = torch.cat(g_chunks, dim=0)
            feat_norm = torch.cat(n_chunks, dim=0)

            # Optional hash centering to improve bit balance per slide
            if self.center_hash:
                h_b = h_b - h_b.mean(dim=0, keepdim=True)

            def normalize(t: torch.Tensor) -> torch.Tensor:
                t_min, t_max = t.min(), t.max()
                if t_max > t_min:
                    return (t - t_min) / (t_max - t_min)
                return torch.zeros_like(t)

            s_norm = normalize(feat_norm)
            s_hash = normalize(h_b.abs().mean(dim=-1))
            s_guide = normalize(guide_b)

            # Optional low-norm filtering to remove background patches
            if 0.0 < self.min_norm_percentile < 1.0 and nv > kb:
                thresh = torch.quantile(s_norm, self.min_norm_percentile)
                keep_mask = s_norm >= thresh
                if keep_mask.sum().item() >= kb:
                    x_b = x_b[keep_mask]
                    h_b = h_b[keep_mask]
                    s_norm = s_norm[keep_mask]
                    s_hash = s_hash[keep_mask]
                    s_guide = s_guide[keep_mask]
                    feat_norm = feat_norm[keep_mask]
                    valid_idx = valid_idx[keep_mask]
                    nv = valid_idx.numel()
                    kb = min(k, nv)

            # Weighted importance score
            w_sum = self.norm_weight + self.hash_weight + self.guide_weight
            temp_norm = self.norm_weight / w_sum
            temp_hash = self.hash_weight / w_sum
            temp_guide = self.guide_weight / w_sum
            importance = temp_norm * s_norm + temp_hash * s_hash + temp_guide * s_guide

            # 2. Candidate pool (Top-K' by importance)
            candidate_k = min(nv, self.candidate_mul * kb)
            top_vals, top_idx = torch.topk(importance, k=candidate_k)

            # 3. Diversity selection within candidate pool
            # Normalize hash for cosine similarity
            h_b = F.normalize(h_b, dim=-1)
            if self.use_binary_hash:
                h_div = torch.sign(h_b)
                h_div = torch.where(h_div == 0, torch.ones_like(h_div), h_div)
            else:
                h_div = h_b
            cand_h = h_div[top_idx]

            if coords is not None and level0_size is not None:
                if isinstance(level0_size, (tuple, list)):
                    width, height = level0_size
                else:
                    width, height = int(level0_size[0]), int(level0_size[1])

                c = coords[b, valid_idx]
                x_norm = (c[:, 0].float() / max(width, 1)).clamp(0.0, 1.0)
                y_norm = (c[:, 1].float() / max(height, 1)).clamp(0.0, 1.0)
                coord_feat = torch.stack(
                    [x_norm, y_norm, torch.sin(math.pi * x_norm), torch.sin(math.pi * y_norm)],
                    dim=-1,
                )
                cand_coord = coord_feat[top_idx]
                combined = torch.cat(
                    [cand_h * self.hash_weight, cand_coord * self.spatial_weight], dim=-1
                )
                cand_emb = F.normalize(combined, dim=-1)
            else:
                cand_emb = cand_h

            first = 0  # top_idx[0] is highest importance
            
            # 3. GPU Vectorized FPS
            # We maintain a boolean mask of selected items or just construct the index list
            selected = torch.zeros((kb,), dtype=torch.long, device=x.device)
            selected[0] = first
            
            # Initialize min_dist (Distance to the set of selected points)
            # Dist = 1 - CosSim
            # We want to pick point that maximizes min_dist
            
            # Calculate distance from 'first' point to all others
            # h_b: [Nv, H], h_b[first]: [H]
            sim = torch.matmul(cand_emb, cand_emb[first])  # [C]
            max_sim = sim.clone()
            max_sim[first] = 1.0

            for i in range(1, kb):
                next_idx = torch.argmin(max_sim).item()
                selected[i] = next_idx

                sim_new = torch.matmul(cand_emb, cand_emb[next_idx])
                max_sim = torch.maximum(max_sim, sim_new)
                max_sim[next_idx] = 1.0

            # 4. Gather results
            sel_idx = valid_idx[top_idx[selected]]
            
            out_x[b, :kb] = x[b, sel_idx]
            if out_mask is not None:
                out_mask[b, :kb] = True
            if out_coords is not None:
                out_coords[b, :kb] = coords[b, sel_idx]

        return out_x, out_mask, out_coords


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def load_selector(
    feat_dim: int,
    active_n_rois: int,
    hash_bits: int,
    hash_guide_alpha: float,
    hash_chunk_size: int,
    hash_rank: int,
    candidate_mul: int,
    min_norm_percentile: float,
    norm_weight: float,
    hash_weight: float,
    guide_weight: float,
    spatial_weight: float,
    center_hash: bool,
    use_binary_hash: bool,
    checkpoint: str | None,
    device: torch.device,
) -> OptimizedDeepHashSelector:
    selector = OptimizedDeepHashSelector(
        dim=feat_dim,
        num_patches=active_n_rois,
        hash_bits=hash_bits,
        guide_alpha=hash_guide_alpha,
        chunk_size=hash_chunk_size,
        hash_rank=hash_rank,
        candidate_mul=candidate_mul,
        min_norm_percentile=min_norm_percentile,
        norm_weight=norm_weight,
        hash_weight=hash_weight,
        guide_weight=guide_weight,
        spatial_weight=spatial_weight,
        center_hash=center_hash,
        use_binary_hash=use_binary_hash,
    )
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("hash_model", ckpt.get("model", ckpt))
        state = _strip_prefix(state, ("module.",))
        selector_state = {}
        for k, v in state.items():
            if k.startswith("encoder.selector."):
                selector_state[k.replace("encoder.selector.", "")] = v
            elif k.startswith("selector."):
                selector_state[k.replace("selector.", "")] = v
        if selector_state:
            # Backward-compat: old checkpoints had hash_proj.{weight,bias}
            if "hash_proj.weight" in selector_state and "hash_proj1.weight" not in selector_state:
                with torch.no_grad():
                    w = selector_state["hash_proj.weight"]  # [hash_bits, dim]
                    # Low-rank factorization via SVD: W ≈ U S Vh
                    # proj2 = U * sqrt(S), proj1 = sqrt(S) * Vh
                    try:
                        u, s, vh = torch.linalg.svd(w, full_matrices=False)
                        r = min(selector.hash_rank, s.numel())
                        u = u[:, :r]
                        s = s[:r]
                        vh = vh[:r, :]
                        s_sqrt = torch.sqrt(s)
                        proj2 = u * s_sqrt.unsqueeze(0)
                        proj1 = (s_sqrt.unsqueeze(1) * vh)
                        selector_state["hash_proj1.weight"] = proj1
                        selector_state["hash_proj2.weight"] = proj2
                    except Exception as e:
                        print(f"Warning: SVD init for low-rank hash failed ({e}); using random init for hash_proj1/2.")
                    # Remove old keys to avoid strict loading failure
                    selector_state.pop("hash_proj.weight", None)
                    selector_state.pop("hash_proj.bias", None)

            selector.load_state_dict(selector_state, strict=False)
        else:
            print("Warning: selector weights not found in checkpoint; using random init.")
    selector.eval()
    selector.to(device)
    return selector


def select_one(
    selector: OptimizedDeepHashSelector,
    pack: Dict[str, torch.Tensor],
    hash_on_mags: List[int],
    max_per_mag: Dict[int, int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    roi_feat = pack["roi_feat"].float().to(device)
    roi_mag = pack["roi_mag"].to(device)
    coords = pack.get("coords_level0")
    mags = pack.get("mags")
    level0_size = pack.get("level0_size")

    if roi_feat.dim() == 3:  # legacy [N,3,D]
        n, m, d = roi_feat.shape
        roi_feat = roi_feat.reshape(n * m, d)
        if roi_mag is None:
            mag_values = mags if mags is not None else list(range(m))
            mag_tensor = torch.tensor(mag_values, dtype=torch.int32)
            roi_mag = mag_tensor.unsqueeze(0).repeat(n, 1).reshape(-1)
        if coords is not None:
            coords = coords.to(torch.int32)
            coords = coords.unsqueeze(1).repeat(1, m, 1).reshape(n * m, 2)

    if roi_mag is None:
        raise ValueError("Feature pack missing 'roi_mag'")

    if coords is not None:
        coords = coords.to(torch.int32).to(device)

    unique_mags = sorted(set(roi_mag.view(-1).tolist()))
    if mags is None:
        mags = unique_mags

    feats_out = []
    mags_out = []
    coords_out = []

    with torch.no_grad():
        for mag in unique_mags:
            mask = roi_mag == mag
            feat_mag = roi_feat[mask]
            coords_mag = coords[mask] if coords is not None else None

            if mag in hash_on_mags:
                # Dynamically adjust selector's K to match the target cap for this mag
                target_k = max_per_mag.get(mag, selector.num_patches)
                original_k = selector.num_patches
                if target_k > 0:
                    selector.num_patches = target_k
                
                x = feat_mag.unsqueeze(0)  # [1, N, D]
                # If we have fewer patches than target_k, selector handles it gracefully (returns all)
                m = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
                c = coords_mag.unsqueeze(0) if coords_mag is not None else None
                
                try:
                    x_sel, _, c_sel = selector(x, m, c, level0_size=level0_size)
                finally:
                    # Restore original K just in case
                    selector.num_patches = original_k
                
                feat_mag = x_sel.squeeze(0)
                coords_mag = c_sel.squeeze(0) if c_sel is not None else None

            cap = max_per_mag.get(mag, -1)
            if cap > 0 and feat_mag.shape[0] > cap:
                perm = torch.randperm(feat_mag.shape[0], device=device)[:cap]
                feat_mag = feat_mag[perm]
                if coords_mag is not None:
                    coords_mag = coords_mag[perm]

            feats_out.append(feat_mag)
            mags_out.append(torch.full((feat_mag.shape[0],), mag, dtype=roi_mag.dtype, device=device))
            if coords_mag is not None:
                coords_out.append(coords_mag)

    roi_feat_new = torch.cat(feats_out, dim=0).cpu()
    roi_mag_new = torch.cat(mags_out, dim=0).cpu()
    coords_new = torch.cat(coords_out, dim=0).cpu() if coords_out else None

    new_pack = dict(pack)
    new_pack["roi_feat"] = roi_feat_new
    new_pack["roi_mag"] = roi_mag_new
    if coords_new is not None:
        new_pack["coords_level0"] = coords_new
    new_pack["mags"] = mags
    return new_pack


def worker(gpu_id: int, files: List[str], args, feat_dim: int, start_idx: int):
    """
    Worker function to process a subset of files on a specific GPU.
    """
    try:
        seed_base = int(getattr(args, "seed", 42))
        if gpu_id is None or gpu_id < 0:
            setup_seed(seed_base)
        else:
            setup_seed(seed_base + int(gpu_id))
        
        # Map logical GPU ID (from multiprocessing) to actual cuda device
        if gpu_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu_id}")
        
        # Load parsing of args from object or dict
        input_dir = args.input_dir
        output_dir = args.output_dir
        active_n_rois = args.active_n_rois
        hash_bits = args.hash_bits
        hash_guide_alpha = args.hash_guide_alpha
        hash_chunk_size = args.hash_chunk_size
        hash_rank = args.hash_rank
        candidate_mul = args.candidate_mul
        min_norm_percentile = args.min_norm_percentile
        norm_weight = args.norm_weight
        hash_weight = args.hash_weight
        guide_weight = args.guide_weight
        spatial_weight = args.spatial_weight
        center_hash = args.center_hash
        use_binary_hash = args.use_binary_hash
        checkpoint = args.checkpoint
        
        # Parse list/dict args
        if isinstance(args.hash_on_mags, list):
            hash_on_mags = args.hash_on_mags
        else:
            hash_on_mags = [int(x) for x in str(args.hash_on_mags).split(",") if x.strip() != ""]
             
        max_per_mag = {}
        if isinstance(args.max_per_mag, dict):
            max_per_mag = args.max_per_mag
        elif args.max_per_mag:
            for item in str(args.max_per_mag).split(","):
                mag_str, cap_str = item.split(":")
                max_per_mag[int(mag_str)] = int(cap_str)

        selector = load_selector(
            feat_dim=feat_dim,
            active_n_rois=active_n_rois,
            hash_bits=hash_bits,
            hash_guide_alpha=hash_guide_alpha,
            hash_chunk_size=hash_chunk_size,
            hash_rank=hash_rank,
            candidate_mul=candidate_mul,
            min_norm_percentile=min_norm_percentile,
            norm_weight=norm_weight,
            hash_weight=hash_weight,
            guide_weight=guide_weight,
            spatial_weight=spatial_weight,
            center_hash=center_hash,
            use_binary_hash=use_binary_hash,
            checkpoint=checkpoint,
            device=device,
        )

        for i, fname in enumerate(files):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            
            try:
                pack = torch.load(in_path, map_location="cpu")
                new_pack = select_one(selector, pack, hash_on_mags, max_per_mag, device)
                torch.save(new_pack, out_path)
            except Exception as e:
                print(f"[GPU{gpu_id}] Error processing {fname}: {e}")

            if (i + 1) % 50 == 0:
                print(f"[GPU{gpu_id}] Processed {i + 1}/{len(files)}")
                
    except Exception as e:
        print(f"[GPU{gpu_id}] Critical worker error: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Offline deep hash selection for WSI patches")
    parser.add_argument("--config", default=None, help="Path to config yaml file")
    # Arguments that can be overridden by config
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--active_n_rois", type=int, default=2048)
    parser.add_argument("--hash_bits", type=int, default=64)
    parser.add_argument("--hash_guide_alpha", type=float, default=0.2)
    parser.add_argument("--hash_chunk_size", type=int, default=1024)
    parser.add_argument("--hash_rank", type=int, default=64)
    parser.add_argument("--candidate_mul", type=int, default=4)
    parser.add_argument("--min_norm_percentile", type=float, default=0.2)
    parser.add_argument("--norm_weight", type=float, default=0.4)
    parser.add_argument("--hash_weight", type=float, default=0.4)
    parser.add_argument("--guide_weight", type=float, default=0.2)
    parser.add_argument("--spatial_weight", type=float, default=0.3)
    parser.add_argument("--center_hash", action="store_true", default=True)
    parser.add_argument("--use_binary_hash", action="store_true", default=True)
    parser.add_argument("--hash_on_mags", default="0")
    parser.add_argument("--max_per_mag", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # Load Config
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if "Data" in cfg:
            if cfg["Data"].get("input_dir"): args.input_dir = cfg["Data"]["input_dir"]
            if cfg["Data"].get("output_dir"): args.output_dir = cfg["Data"]["output_dir"]
        
        if "Model" in cfg:
            m = cfg["Model"]
            if "checkpoint" in m: args.checkpoint = m["checkpoint"]
            if "active_n_rois" in m: args.active_n_rois = m["active_n_rois"]
            if "hash_bits" in m: args.hash_bits = m["hash_bits"]
            if "hash_guide_alpha" in m: args.hash_guide_alpha = m["hash_guide_alpha"]
            if "hash_chunk_size" in m: args.hash_chunk_size = m["hash_chunk_size"]
            if "hash_rank" in m: args.hash_rank = m["hash_rank"]
            if "candidate_mul" in m: args.candidate_mul = m["candidate_mul"]
            if "min_norm_percentile" in m: args.min_norm_percentile = m["min_norm_percentile"]
            if "norm_weight" in m: args.norm_weight = m["norm_weight"]
            if "hash_weight" in m: args.hash_weight = m["hash_weight"]
            if "guide_weight" in m: args.guide_weight = m["guide_weight"]
            if "spatial_weight" in m: args.spatial_weight = m["spatial_weight"]
            if "center_hash" in m: args.center_hash = m["center_hash"]
            if "use_binary_hash" in m: args.use_binary_hash = m["use_binary_hash"]
            if "hash_on_mags" in m: args.hash_on_mags = m["hash_on_mags"]
            if "max_per_mag" in m: args.max_per_mag = m["max_per_mag"]

    if args.input_dir is None or args.output_dir is None:
        parser.error("input_dir and output_dir required (either via CLI or Config)")

    setup_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Probe a sample to get feature dim
    all_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pt")]
    if not all_files:
        print(f"No .pt files found in {args.input_dir}")
        return

    sample_path = os.path.join(args.input_dir, all_files[0])
    sample_pack = torch.load(sample_path, map_location="cpu")
    feat_dim = int(sample_pack["roi_feat"].shape[-1])
    print(f"Detected feature dim: {feat_dim}, Found {len(all_files)} files.")

    # Multi-GPU Setup
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs.")
    else:
        n_gpus = 0
        print("No GPU found, running CPU.")

    if n_gpus > 1:
        # Split files across GPUs
        chunk_size = (len(all_files) + n_gpus - 1) // n_gpus
        processes = []
        mp.set_start_method('spawn', force=True)
        
        print(f"Spawning {n_gpus} processes...")
        
        for rank in range(n_gpus):
            start = rank * chunk_size
            end = min(start + chunk_size, len(all_files))
            files_chunk = all_files[start:end]
            
            p = mp.Process(
                target=worker, 
                args=(rank, files_chunk, args, feat_dim, start)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    else:
        # Single Process (CPU or Single GPU)
        device_str = "cuda:0" if n_gpus > 0 else "cpu"
        device = torch.device(device_str)
        worker(0 if n_gpus > 0 else -1, all_files, args, feat_dim, 0)

    print("Done. Output:", args.output_dir)


if __name__ == "__main__":
    main()
