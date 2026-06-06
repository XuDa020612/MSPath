import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttnNetGated(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.attn_a = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attn_b = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_c = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        a = self.attn_a(x)
        b = self.attn_b(x)
        A = self.attn_c(a * b)  # [B,N,1]
        return A


class WSIAttentionPool(nn.Module):
    """CHIEF-style attention pooling over ROI tokens.

    Input:  tokens [B, N, D]
    Output:
      - global [B, D]
      - attn_raw [B, N]
      - topk_tokens [B, K, D]
      - topk_idx [B, K]
    """

    def __init__(self, in_dim: int = 768, proj_dim: int = 512, attn_hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn = AttnNetGated(proj_dim, attn_hidden, dropout=dropout)
        self.out_dim = in_dim

    def forward(self, tokens: torch.Tensor, topk: int = 64, mask: Optional[torch.Tensor] = None):
        """tokens: [B,N,D], mask: [B,N] bool where True indicates valid."""
        h = self.proj(tokens)  # [B,N,proj]
        a = self.attn(h).squeeze(-1)  # [B,N]

        if mask is not None:
            # Set invalid positions to -inf so they get ~0 probability.
            a_masked = a.masked_fill(~mask, float("-inf"))
            valid = mask.any(dim=1, keepdim=True)
            a_masked = a_masked.masked_fill(~valid, 0.0)
            attn = F.softmax(a_masked, dim=1)
            attn = attn.masked_fill(~valid, 0.0)
        else:
            attn = F.softmax(a, dim=1)

        global_vec = torch.einsum("bn,bnd->bd", attn, tokens)  # weighted sum in original space

        n = tokens.shape[1]
        k = min(topk, n)
        # Ensure we don't pick padded tokens.
        score = attn
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        topk_idx = torch.topk(score, k=k, dim=1).indices  # [B,K]
        b_idx = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
        topk_tokens = tokens[b_idx, topk_idx]  # [B,K,D]
        return {
            "global": global_vec,
            "attn_raw": a,
            "attn": attn,
            "topk_tokens": topk_tokens,
            "topk_idx": topk_idx,
        }
