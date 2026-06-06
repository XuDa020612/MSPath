from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict

from .mil_pool import WSIAttentionPool


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        def _inner_forward(x_in):
            # x_in: [B, N, D]
            # key_padding_mask: [B, N] (True where padded)
            # nn.MultiheadAttention expects key_padding_mask=True for IGNORED positions
            # We assume 'mask' is 1 for VALID, 0 for PADDED. -> key_padding_mask = ~mask.bool()
            key_padding_mask = None
            if mask is not None:
                # mask is [B, N], 1=valid, 0=pad
                # MultiheadAttention mask: True for values to be ignored
                key_padding_mask = mask.to(torch.bool) == False
            
            attn_out, _ = self.attn(x_in, x_in, x_in, key_padding_mask=key_padding_mask, need_weights=False)
            return attn_out

        if self.training and x.requires_grad:
            # Gradient checkpointing could be added here if needed
            x = x + _inner_forward(self.ln1(x))
        else:
            x = x + _inner_forward(self.ln1(x))
        
        x = x + self.mlp(self.ln2(x))
        return x


class _CoordinateEncoder(nn.Module):
    """Map level-0 coordinates to learnable positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        level0_size: torch.Tensor,
    ) -> torch.Tensor:
        # coords: [B,N,2] in level0 pixels, level0_size: [B,2]
        b, n, _ = coords.shape
        device = coords.device
        size = level0_size.to(device=device, dtype=torch.float32).clamp(min=1.0)
        width = size[:, 0].unsqueeze(1)
        height = size[:, 1].unsqueeze(1)

        x = coords[..., 0].to(torch.float32) / width.clamp(min=1.0)
        y = coords[..., 1].to(torch.float32) / height.clamp(min=1.0)
        feat = torch.stack(
            [
                x.clamp(0.0, 1.0),
                y.clamp(0.0, 1.0),
                torch.sin(math.pi * x),
                torch.sin(math.pi * y),
            ],
            dim=-1,
        )
        pos = self.net(feat)
        if mask is not None:
            # Zero out positions for masked items
            pos = pos * mask.unsqueeze(-1)
        return pos


class DeepHashSelector(nn.Module):
    """Low-rank deep hashing selector with lightweight task-guided diversity sampling."""

    def __init__(
        self,
        dim: int,
        num_patches: int = 2048,
        hash_bits: int = 64,
        guide_alpha: float = 0.2,
        chunk_size: int = 1024,
        hash_rank: int = 64,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.hash_bits = hash_bits
        self.guide_alpha = guide_alpha
        self.chunk_size = chunk_size
        self.hash_rank = max(1, min(int(hash_rank), dim))
        self.hash_proj1 = nn.Linear(dim, self.hash_rank, bias=False)
        self.hash_proj2 = nn.Linear(self.hash_rank, hash_bits, bias=False)
        self.guide_proj = nn.Linear(dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        level0_size: Optional[torch.Tensor] = None
    ):
        if self.num_patches <= 0 or x.shape[1] <= self.num_patches:
            return x, mask, coords

        bsz, n, d = x.shape
        k = self.num_patches

        out_x = x.new_empty((bsz, k, d), device=x.device, dtype=x.dtype)
        out_mask = None
        if mask is not None:
            out_mask = mask.new_zeros((bsz, k), device=x.device, dtype=mask.dtype)

        out_coords = None
        if coords is not None:
            out_coords = coords.new_zeros((bsz, k, 2), device=x.device, dtype=coords.dtype)

        for b in range(bsz):
            # Select valid indices
            if mask is not None:
                valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            else:
                valid_idx = torch.arange(n, device=x.device)
            
            nv = valid_idx.numel()
            if nv == 0:
                continue
                
            kb = min(k, nv)
            
            # Extract valid features
            x_b = x[b, valid_idx]  # [nv, D]
            
            # 1. Project to Hash
            h_b = torch.tanh(self.hash_proj2(self.hash_proj1(x_b)))
            
            # 2. Guide score
            g_b = self.guide_proj(x_b).squeeze(-1)
            
            # Implementation: Task Enhanced Random Sampling
            weights = torch.softmax(g_b, dim=0)
            if kb < nv:
                sel_indices = torch.multinomial(weights, kb, replacement=False)
            else:
                sel_indices = torch.arange(nv, device=x.device)
            
            sel_global_idx = valid_idx[sel_indices]
            
            out_x[b, :kb] = x[b, sel_global_idx]
            if out_mask is not None:
                # out_mask was initialized to zeros, set valid entries to 1
                out_mask[b, :kb] = 1
            if out_coords is not None:
                out_coords[b, :kb] = coords[b, sel_global_idx]

        return out_x, out_mask, out_coords


@dataclass
class WSIEncoderConfig:
    tile_dim: int = 768
    roi_dim: int = 768
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.25
    topk: int = 64
    mil_hidden: int = 256
    bottleneck_num_latents: int = 64
    active_n_rois: int = 2048
    hash_bits: int = 64
    hash_rank: int = 64
    hash_guide_alpha: float = 0.2
    hash_on_mag: int | list[int] = 0
    hash_chunk_size: int = 1024


class PerceiverCrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, return_attn: bool = False) -> torch.Tensor:
        # === [MODIFIED for Visualization] ===
        # 原本只返回out，现在可选返回注意力权重 attn_weights
        q_in = self.ln_q(q)
        kv_in = self.ln_kv(kv)
        attn_out, attn_weights = self.attn(q_in, kv_in, kv_in, key_padding_mask=key_padding_mask, need_weights=return_attn)
        out = q + attn_out if self.use_residual else attn_out
        out = out + self.mlp(self.ln_out(out))
        if return_attn:
            return out, attn_weights
        return out
        # === [END MODIFIED] ===


class WSIEncoder(nn.Module):
    """Hierarchical WSI encoder with U-Net style fusion and bottleneck reconstruction."""

    def __init__(self, cfg: WSIEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.selector = DeepHashSelector(
            cfg.tile_dim,
            cfg.active_n_rois,
            cfg.hash_bits,
            cfg.hash_guide_alpha,
            cfg.hash_chunk_size,
            cfg.hash_rank,
        )
        self.input_proj = nn.Linear(cfg.tile_dim, cfg.roi_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.roi_dim) * 0.02)

        self.coord_enc = _CoordinateEncoder(cfg.roi_dim)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(cfg.roi_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.mil = WSIAttentionPool(
            in_dim=cfg.roi_dim,
            proj_dim=max(cfg.roi_dim // 2, 256),
            attn_hidden=cfg.mil_hidden,
            dropout=cfg.dropout,
        )

        self.down_high_to_mid = PerceiverCrossAttn(cfg.roi_dim, cfg.num_heads, cfg.dropout)
        self.down_mid_to_low = PerceiverCrossAttn(cfg.roi_dim, cfg.num_heads, cfg.dropout)

        self.bottleneck_query = nn.Parameter(torch.randn(1, cfg.bottleneck_num_latents, cfg.roi_dim) * 0.02)
        self.to_bottleneck = PerceiverCrossAttn(cfg.roi_dim, cfg.num_heads, cfg.dropout)

        self.up_to_low = PerceiverCrossAttn(cfg.roi_dim, cfg.num_heads, cfg.dropout)
        self.up_to_mid = PerceiverCrossAttn(cfg.roi_dim, cfg.num_heads, cfg.dropout)
        self.up_to_high = PerceiverCrossAttn(cfg.roi_dim, cfg.num_heads, cfg.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        features_by_mag: Dict[int, torch.Tensor],
        masks_by_mag: Dict[int, torch.Tensor],
        coords_by_mag: Optional[Dict[int, torch.Tensor]] = None,
        level0_size: Optional[torch.Tensor] = None,
        return_cross_attn: bool = False, # === [MODIFIED for Visualization] ===
    ) -> dict:
        device = next(iter(features_by_mag.values())).device
        bsz = next(iter(features_by_mag.values())).shape[0]

        encoded_scales = {}
        per_scale_out = {}
        scale_topk = []

        hash_on = self.cfg.hash_on_mag
        if isinstance(hash_on, int):
            hash_on = [hash_on]

        # 1. ENCODE EACH SCALE
        for mag in features_by_mag:
            feat = features_by_mag[mag]
            mask = masks_by_mag[mag]
            coords = coords_by_mag[mag] if coords_by_mag else None

            # Optional: Select subset of patches using Hashing for high mags
            if mag in hash_on:
                feat, mask, coords = self.selector(feat, mask, coords, level0_size=level0_size)

            encoded = self._encode_scale(feat, mask, coords, level0_size)
            encoded_scales[mag] = encoded
            per_scale_out[mag] = encoded
            scale_topk.append(encoded["topk_tokens"])

        # 2. HIERARCHICAL FUSION (U-Net style)
        has_0 = 0 in encoded_scales
        has_1 = 1 in encoded_scales
        has_2 = 2 in encoded_scales

        f0 = encoded_scales[0]["features"] if has_0 else None
        m0 = encoded_scales[0]["mask"] if has_0 else None

        f1 = encoded_scales[1]["features"] if has_1 else None
        m1 = encoded_scales[1]["mask"] if has_1 else None

        f2 = encoded_scales[2]["features"] if has_2 else None
        m2 = encoded_scales[2]["mask"] if has_2 else None

        # === [MODIFIED for Visualization] ===
        cross_attns = {}
        # Downward path
        if has_0 and has_1:
            mask_0 = m0 == 0 # True for padding
            if return_cross_attn:
                f_mid_fused, attn_1_0 = self.down_high_to_mid(f1, f0, key_padding_mask=mask_0, return_attn=True)
                cross_attns["1_to_0"] = attn_1_0
            else:
                f_mid_fused = self.down_high_to_mid(f1, f0, key_padding_mask=mask_0)
        elif has_1:
            f_mid_fused = f1
        else:
            f_mid_fused = None

        if has_1 and has_2 and f_mid_fused is not None:
            mask_1 = m1 == 0
            if return_cross_attn:
                f_low_fused, attn_2_1 = self.down_mid_to_low(f2, f_mid_fused, key_padding_mask=mask_1, return_attn=True)
                cross_attns["2_to_1"] = attn_2_1
            else:
                f_low_fused = self.down_mid_to_low(f2, f_mid_fused, key_padding_mask=mask_1)
        elif has_2:
            f_low_fused = f2
        else:
            f_low_fused = f_mid_fused if f_mid_fused is not None else (f0 if has_0 else None)
        # === [END MODIFIED] ===

        # Bottleneck
        latents = self.bottleneck_query.expand(bsz, -1, -1)
        if f_low_fused is not None:
            src_mask = None
            if has_2: src_mask = m2 == 0
            elif has_1: src_mask = m1 == 0
            elif has_0: src_mask = m0 == 0
            
            bottleneck_out = self.to_bottleneck(latents, f_low_fused, key_padding_mask=src_mask)
        else:
            bottleneck_out = latents 

        bottleneck_cls = bottleneck_out.mean(dim=1)

        # Reconstruction (Upward path)
        aux_preds = {}
        aux_targets = {}

        if has_2:
            rec_2 = self.up_to_low(f2, bottleneck_out)
            aux_preds[2] = rec_2
            aux_targets[2] = f2.detach()
        else:
            rec_2 = None

        if has_1:
            rec_1 = self.up_to_mid(f1, bottleneck_out)
            aux_preds[1] = rec_1
            aux_targets[1] = f1.detach()
        else:
            rec_1 = None

        if has_0:
            rec_0 = self.up_to_high(f0, bottleneck_out)
            aux_preds[0] = rec_0
            aux_targets[0] = f0.detach()
        else:
            rec_0 = None

        if has_0:
            recon_highest = rec_0
        elif has_1:
            recon_highest = rec_1
        elif has_2:
            recon_highest = rec_2
        else:
            recon_highest = torch.empty(bsz, 0, self.cfg.roi_dim, device=device)

        fused_topk = torch.cat(scale_topk, dim=1) if scale_topk else torch.empty(bsz, 0, self.cfg.roi_dim, device=device)

        return {
            "global": bottleneck_cls,
            "scale_tokens": None,
            "per_scale": per_scale_out,
            "topk_tokens": fused_topk,
            "recon_highest": recon_highest,
            "aux_preds": aux_preds,
            "aux_targets": aux_targets,
            "cross_attns": cross_attns, # === [MODIFIED for Visualization] ===
        }

    def _encode_scale(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
        coords: Optional[torch.Tensor],
        level0_size: Optional[torch.Tensor],
    ) -> dict:
        tokens = self.input_proj(feat)
        if coords is not None and level0_size is not None:
            pos = self.coord_enc(coords, mask, level0_size)
            tokens = tokens + pos

        bsz = tokens.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        tokens_with_cls = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((bsz, 1), device=mask.device, dtype=mask.dtype)
        mask_with_cls = torch.cat([cls_mask, mask], dim=1)

        for blk in self.blocks:
            tokens_with_cls = blk(tokens_with_cls, mask=mask_with_cls)

        cls_out = tokens_with_cls[:, 0]
        valid = mask.any(dim=1, keepdim=True).to(dtype=cls_out.dtype)
        cls_out = cls_out * valid

        body_tokens = tokens_with_cls[:, 1:]

        pooled = self.mil(body_tokens, topk=self.cfg.topk, mask=mask)
        mil_global = pooled["global"]
        pooled["features"] = body_tokens
        pooled["coords"] = coords
        pooled["mask"] = mask 
        pooled["global_cls"] = cls_out
        pooled["global_mil"] = mil_global
        return pooled

