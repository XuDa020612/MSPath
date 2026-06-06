from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cross_modal import CrossModalAdapter
from .resampler import VisualResampler
from .wsi_encoder import WSIEncoder, WSIEncoderConfig


@dataclass
class CoCaConfig:
    tile_dim: int = 768
    roi_dim: int = 768
    proj_dim: int = 512
    mil_hidden: int = 256
    topk: int = 64
    prefix_len: int = 32
    encoder_depth: int = 2
    encoder_heads: int = 8
    encoder_dropout: float = 0.25
    mag_embed_scale: float = 20.0
    bottleneck_num_latents: int = 64
    active_n_rois: int = 2048
    hash_bits: int = 64
    hash_rank: int = 64
    hash_guide_alpha: float = 0.4
    hash_on_mag: int | list[int] = 0
    hash_chunk_size: int = 1024


class WSIReportCoCa(nn.Module):
    """CoCa-style model head (visual only).

    Expects pre-extracted ROI tokens [B,N,768] plus magnification labels.
    Produces:
      - z_img for retrieval (normalized)
      - visual prefix embeddings for LLM generation
    """

    def __init__(self, llm_dim: int, cfg: CoCaConfig):
        super().__init__()
        self.cfg = cfg
        encoder_cfg = WSIEncoderConfig(
            tile_dim=cfg.tile_dim,
            roi_dim=cfg.roi_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.encoder_heads,
            dropout=cfg.encoder_dropout,
            topk=cfg.topk,
            mil_hidden=cfg.mil_hidden,
            bottleneck_num_latents=cfg.bottleneck_num_latents,
            active_n_rois=cfg.active_n_rois,
            hash_bits=cfg.hash_bits,
            hash_rank=cfg.hash_rank,
            hash_guide_alpha=cfg.hash_guide_alpha,
            hash_on_mag=cfg.hash_on_mag,
            hash_chunk_size=cfg.hash_chunk_size,
        )
        self.encoder = WSIEncoder(encoder_cfg)
        self.img_proj = nn.Linear(cfg.roi_dim, cfg.proj_dim)
        # Learnable temperature for contrastive learning (CLIP-style).
        # Initialized to log(1/0.07) so exp(logit_scale) ~= 14.285.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.cross_modal = CrossModalAdapter(cfg.roi_dim, llm_dim)
        self.resampler = VisualResampler(
            visual_dim=cfg.roi_dim,
            llm_dim=llm_dim,
            prefix_len=cfg.prefix_len,
            instruction_dim=llm_dim,
        )

    def forward(
        self,
        features_by_mag: Dict[int, torch.Tensor],
        masks_by_mag: Dict[int, torch.Tensor],
        coords_by_mag: Optional[Dict[int, torch.Tensor]] = None,
        level0_size: Optional[torch.Tensor] = None,
        instruction_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(
            features_by_mag,
            masks_by_mag=masks_by_mag,
            coords_by_mag=coords_by_mag,
            level0_size=level0_size,
        )
        g = enc["global"]  # [B,roi_dim]
        z_img = torch.nn.functional.normalize(self.img_proj(g), dim=-1)
        
        # Calculate Aux Reconstruction Loss
        aux_preds = enc.get("aux_preds", {})
        aux_targets = enc.get("aux_targets", {})
        per_scale = enc.get("per_scale", {})
        
        loss_recon = torch.tensor(0.0, device=g.device)
        count = 0
        
        for k in aux_preds:
            if k in aux_targets and k in per_scale:
                pred = aux_preds[k]
                target = aux_targets[k].detach() # Detached to prevent encoder collapse
                mask = per_scale[k]["mask"]
                
                sq_diff = (pred - target) ** 2
                if mask is not None:
                    # mask: [B, N] -> [B, N, 1]
                    m = mask.unsqueeze(-1).float()
                    masked_sq_diff = sq_diff * m
                    loss_k = masked_sq_diff.sum() / (m.sum() * target.shape[-1] + 1e-6)
                else:
                    loss_k = sq_diff.mean()
                
                loss_recon = loss_recon + loss_k
                count += 1
        
        if count > 0:
            loss_recon = loss_recon / count

        cross_tokens = self.cross_modal(enc["topk_tokens"], instruction_tokens)

        # Note: resampler uses cross_tokens as the "visual" source context (K/V)
        prefix = self.resampler(cross_tokens, instruction_tokens=instruction_tokens)
        return {
            "z_img": z_img,
            "prefix": prefix,
            "scale_tokens": enc.get("scale_tokens"),
            "per_scale": enc.get("per_scale"),
            "loss_recon": loss_recon,
        }
