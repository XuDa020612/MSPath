import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from transformers import AutoModel, AutoTokenizer

from .wsi_encoder import WSIEncoder, WSIEncoderConfig


class ABMIL_Attention(nn.Module):
    def __init__(self, L=768, D=256, K=1, dropout=False):
        super(ABMIL_Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
        )

    def forward(self, x):
        A = self.attention(x)
        A = torch.softmax(A, dim=1)
        M = torch.bmm(A.transpose(1, 2), x)
        return M.squeeze(1)

class WSIDownstreamModel(nn.Module):
    def __init__(self, encoder_cfg: WSIEncoderConfig, task_type="stage", num_classes=4, tune_strategy="frozen", text_encoder_name=None):
        super().__init__()
        
        # 文本编码器（永久冻结）
        self.use_text = (text_encoder_name is not None and text_encoder_name != "")
        if self.use_text:
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_dim = self.text_encoder.config.hidden_size
        else:
            self.text_dim = 0

        # WSI编码器
        self.wsi_encoder = WSIEncoder(encoder_cfg)
        # 下游MIL层（🔥 强制可训练，核心修复）
        self.recon_mil = ABMIL_Attention(L=encoder_cfg.roi_dim, D=256, K=1)

        # ===================== 核心冻结策略 =====================
        # 1. frozen：仅冻结WSI编码器，下游层(recon_mil+head)全部可训练
        # 2. partial/finetune：部分解冻编码器，下游层全部可训练
        # ======================================================
        for param in self.wsi_encoder.parameters():
            param.requires_grad = False  # 默认全冻结

        if tune_strategy == "partial":
            # 仅解冻编码器顶层
            for name, param in self.wsi_encoder.named_parameters():
                if "mil" in name or "down_high_to_mid" in name:
                    param.requires_grad = True
        elif tune_strategy == "finetune":
            # 全量微调编码器
            for param in self.wsi_encoder.parameters():
                param.requires_grad = True

        # 任务头
        fusion_dim = encoder_cfg.roi_dim + self.text_dim
        if task_type == "stage":
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(fusion_dim, fusion_dim//2),
                nn.LayerNorm(fusion_dim//2),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(fusion_dim//2, fusion_dim//4),
                nn.LayerNorm(fusion_dim//4),
                nn.GELU(),
                nn.Linear(fusion_dim//4, num_classes)
            )
        elif task_type == "survival":
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(fusion_dim, fusion_dim//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim//2, 1)
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # 🔥 强制下游层永远可训练（绝对不能冻结！）
        for param in self.recon_mil.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def load_pretrained_encoder(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if k.startswith("wsi_encoder."):
                new_state_dict[k.replace("wsi_encoder.", "")] = v
                
        self.wsi_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Successfully loaded pretrained encoder from {checkpoint_path}")

    def forward(
        self,
        features_by_mag: Dict[int, torch.Tensor],
        masks_by_mag: Dict[int, torch.Tensor],
        coords_by_mag: Optional[Dict[int, torch.Tensor]] = None,
        level0_size: Optional[torch.Tensor] = None,
        texts: Optional[list] = None,
    ):
        # 🔥 绝对删除 torch.no_grad() / eval()，保留完整计算图
        out = self.wsi_encoder(
            features_by_mag=features_by_mag,
            masks_by_mag=masks_by_mag,
            coords_by_mag=coords_by_mag,
            level0_size=level0_size
        )
            
        # 特征聚合
        aux_keys = list(out.get("aux_preds", {}).keys())
        if aux_keys:
            lowest_mag = max(aux_keys)
            global_feat = self.recon_mil(out["aux_preds"][lowest_mag])
        else:
            global_feat = out["global"]
        
        # 文本融合
        if self.use_text and texts is not None:
            device = global_feat.device
            encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                text_feat = self.text_encoder(**encoded).last_hidden_state[:, 0, :]
            fusion_feat = torch.cat([global_feat, text_feat], dim=-1)
        else:
            fusion_feat = global_feat
            
        preds = self.head(fusion_feat)
        return preds