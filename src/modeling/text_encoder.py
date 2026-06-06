from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HFTextEncoder(nn.Module):
    """Thin wrapper around a HuggingFace encoder model for retrieval alignment.

    This is intentionally separate from the frozen Qwen used for generation.
    """

    def __init__(self, backbone, hidden_size: int, proj_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        # Trainable CLS query used to pool token hidden states into a single global vector.
        self.cls_query = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.proj = nn.Linear(hidden_size, proj_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Prefer CLS if available; otherwise mean-pool.
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
        else:
            h = out[0]

        # Trainable query pooling over tokens (masked).
        # h: [B,T,H], attention_mask: [B,T]
        q = self.cls_query.to(device=h.device, dtype=h.dtype).squeeze(0)  # [H]
        scores = torch.einsum("bth,h->bt", h, q) / math.sqrt(h.shape[-1])
        if attention_mask is not None:
            attn_mask = attention_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~attn_mask, -1e4)
            valid = attn_mask.any(dim=1)  # [B]
        else:
            attn_mask = None
            valid = torch.ones(h.shape[0], device=h.device, dtype=torch.bool)

        attn = F.softmax(scores, dim=1)
        if attn_mask is not None:
            attn = attn * attn_mask.to(attn.dtype)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attn = attn * valid.unsqueeze(1).to(attn.dtype)
        pooled = torch.einsum("bt,bth->bh", attn, h)

        z = self.proj(pooled)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z
