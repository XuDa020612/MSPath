from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class CrossModalAdapter(nn.Module):
    """Lightweight cross-attention block to inject instruction tokens into visual tokens."""

    def __init__(self, visual_dim: int, instruction_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.inst_proj = nn.Linear(instruction_dim, visual_dim)
        self.ln_v = nn.LayerNorm(visual_dim)
        self.ln_i = nn.LayerNorm(visual_dim)
        self.attn = nn.MultiheadAttention(visual_dim, num_heads, dropout=dropout, batch_first=True)
        hidden = visual_dim * 4
        self.mlp = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, visual_dim),
        )

    def forward(self, visual_tokens: torch.Tensor, instruction_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if instruction_tokens is None:
            return visual_tokens

        # Keep dtypes consistent across projected instruction tokens and visual tokens.
        # In this repo, LLM embeddings are often fp16 while this module is fp32 by default.
        compute_dtype = self.inst_proj.weight.dtype
        if instruction_tokens.dtype != compute_dtype:
            instruction_tokens = instruction_tokens.to(dtype=compute_dtype)
        if visual_tokens.dtype != compute_dtype:
            visual_tokens = visual_tokens.to(dtype=compute_dtype)
        
        # Project instruction to visual dim
        inst = self.inst_proj(instruction_tokens)
        
        # Modified Fusion: Instruction as Query, Image as Key/Value
        # Q = Instruction (normalized)
        q = self.ln_i(inst)
        # K, V = Visual (normalized)
        k = self.ln_v(visual_tokens)
        
        # Attention(Q, K, V)
        fused, _ = self.attn(q, k, k, need_weights=False)
        
        # Residual Connection to Instruction (Query)
        x = inst + fused
        x = x + self.mlp(x)
        return x
