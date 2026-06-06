import torch
import torch.nn as nn


class ROIFusionMLP(nn.Module):
    """Fuse multi-scale ROI features into a single ROI token.

    Input:  roi_feat [B, N, 3, D]
    Output: roi_tok  [B, N, D_out]
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 768, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        b, n, m, d = roi_feat.shape
        assert m == 3, "Expected 3 magnifications"
        x = roi_feat.reshape(b, n, m * d)
        return self.net(x)
