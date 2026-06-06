import torch
import torch.nn as nn
from typing import Optional


class VisualResampler(nn.Module):
    """Perceiver-style resampler with optional instruction conditioning."""

    def __init__(
        self,
        visual_dim: int,
        llm_dim: int,
        prefix_len: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        instruction_dim: int = None,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.query = nn.Parameter(torch.randn(prefix_len, llm_dim) * 0.02)
        self.kv_proj = nn.Linear(visual_dim, llm_dim)
        self.inst_proj = None
        if instruction_dim is not None:
            self.inst_proj = nn.Linear(instruction_dim, llm_dim) if instruction_dim != llm_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(llm_dim)
        self.ln_kv = nn.LayerNorm(llm_dim)
        self.inst_norm = nn.LayerNorm(llm_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 4, llm_dim),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        instruction_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = visual_tokens.shape[0]
        kv_list = [self.ln_kv(self.kv_proj(visual_tokens))]
        if instruction_tokens is not None and self.inst_proj is not None:
            inst = instruction_tokens
            if inst.dim() == 2:
                inst = inst.unsqueeze(0).expand(b, -1, -1)
            inst = self.inst_proj(inst)
            inst = self.inst_norm(inst)
            kv_list.append(inst)
        kv = torch.cat(kv_list, dim=1)

        q = self.query.unsqueeze(0).expand(b, -1, -1)
        q = self.ln_q(q)
        y, _ = self.attn(q, kv, kv, need_weights=False)
        y = y + self.mlp(y)
        return y
