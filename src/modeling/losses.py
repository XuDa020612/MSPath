import math

import torch
import torch.nn.functional as F


def clip_contrastive_loss_learnable(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    logit_scale: torch.Tensor,
    clamp_min: float = 0.01,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    """Symmetric InfoNCE over a batch with learnable temperature.

    Uses CLIP-style parameterization: logits = exp(logit_scale) * (z_img @ z_txt.T)

    z_img: [B,D] normalized
    z_txt: [B,D] normalized
    logit_scale: scalar tensor (typically a nn.Parameter in the model)
    """
    # Clamp in log-space to keep exp stable.
    logit_scale_clamped = logit_scale.clamp(min=math.log(clamp_min), max=math.log(clamp_max))
    scale = logit_scale_clamped.exp()
    logits = (z_img @ z_txt.t()) * scale
    labels = torch.arange(z_img.shape[0], device=z_img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) * 0.5


def clip_contrastive_loss(z_img: torch.Tensor, z_txt: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE over a batch.

    z_img: [B,D] normalized
    z_txt: [B,D] normalized
    """
    logits = (z_img @ z_txt.t()) / temperature
    labels = torch.arange(z_img.shape[0], device=z_img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) * 0.5
