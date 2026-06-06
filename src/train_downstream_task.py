import math
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
)

# Add src to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.dataset import ROIFeatureDataset, collate_pairs
from src.modeling.coca_model import CoCaConfig
from src.modeling.wsi_encoder import WSIEncoderConfig
from src.modeling.downstream_model import WSIDownstreamModel

def read_yaml_local(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class DownstreamDataset(ROIFeatureDataset):
    # 🔥 修复1：新增task参数，数据集层提前过滤无效样本，避免空批次
    def __init__(self, csv_path: str, feature_dir: str, task: str, **kwargs):
        super().__init__(csv_path, feature_dir, **kwargs)
        self.df = pd.read_csv(csv_path)
        self.task = task

        # 🔥 核心修复：提前过滤无效标签，从源头避免无效数据
        if self.task == "stage":
            self.df = self.df[self.df["stage_label"] != -1].dropna(subset=["stage_label"])
        else:
            self.df = self.df[
                (self.df["surv_time"] != -1.0) & (self.df["surv_status"] != -1)
            ].dropna(subset=["surv_time", "surv_status"])

        # Create mapping for fast lookup
        self.label_map = {}
        for _, row in self.df.iterrows():
            stage_label = row.get("stage_label", -1)
            surv_time = row.get("surv_time", -1.0)
            surv_status = row.get("surv_status", -1)
            
            self.label_map[str(row["case_id"])] = {
                "stage_label": int(stage_label),
                "surv_time": float(surv_time),
                "surv_status": int(surv_status)
            }

    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        case_id = data["case_id"]
        labels = self.label_map.get(case_id, {"stage_label": -1, "surv_time": -1.0, "surv_status": -1})
        data.update(labels)
        return data

def collate_downstream(batch):
    out = collate_pairs(batch)
    out["stage_label"] = torch.tensor([b["stage_label"] for b in batch], dtype=torch.long)
    out["surv_time"] = torch.tensor([b["surv_time"] for b in batch], dtype=torch.float32)
    out["surv_status"] = torch.tensor([b["surv_status"] for b in batch], dtype=torch.float32)
    return out

# 🔥 修复2：标准Cox PH损失实现（生存任务核心修复）
def cox_ph_loss(risk_scores, surv_time, surv_status):
    """
    Compute Standard Cox Proportional Hazards Loss (CORRECT IMPLEMENTATION)
    risk_scores: [B, 1]
    surv_time: [B]
    surv_status: [B] (1=event, 0=censored)
    """
    risk_scores = risk_scores.squeeze(-1)
    event = surv_status.bool()

    # 必须按生存时间降序排序
    sorted_time, idx = torch.sort(surv_time, descending=True)
    sorted_risk = risk_scores[idx]
    sorted_event = event[idx]

    # 向量化计算风险集累计和，无大矩阵，显存高效
    exp_risk = torch.exp(sorted_risk)
    risk_sum = torch.cumsum(exp_risk, dim=0)
    log_risk = torch.log(risk_sum.clamp(min=1e-9))

    # 仅对事件样本计算损失
    loss = - (sorted_risk - log_risk)[sorted_event]

    # 无事件时返回0
    if loss.numel() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
    return loss.mean()

def build_survival_struct(times, events):
    return np.array(
        [(bool(e), float(t)) for e, t in zip(events, times)],
        dtype=[("Status", "?"), ("Survival_in_days", "<f8")],
    )

# 🔥 修复3：使用IPCW加权C-index（生存分析标准指标，适配截尾数据）
def compute_survival_metrics(train_times, train_events, train_scores, test_times, test_events, test_scores):
    try:
        train_y = build_survival_struct(train_times, train_events)
        test_y = build_survival_struct(test_times, test_events)
        
        # 核心：IPCW C-index（替代原始无偏C-index，临床/学术标准）
        c_index = concordance_index_ipcw(train_y, test_y, test_scores)[0]
    except Exception as e:
        print(f"Error calculating C-Index: {e}")
        c_index = 0.0

    return {
        "c_index": float(c_index)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", type=str, choices=["stage", "survival"], required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tune_strategy", type=str, choices=["frozen", "partial", "finetune"], default="frozen")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--train_csv", type=str, default=None, help="Override the training split CSV path")
    parser.add_argument("--val_csv", type=str, default=None, help="Override the validation split CSV path")
    parser.add_argument("--test_csv", type=str, default=None, help="Override the test CSV path for evaluation")
    parser.add_argument(
        "--use_clinical_text",
        action="store_true",
        help="If set, use clinical_history text branch; default is OFF to avoid potential leakage.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.task.upper()} TASK] Device: {device}, Strategy: {args.tune_strategy}")

    cfg = read_yaml_local(args.config)
    data_cfg = cfg["Data"]
    model_cfg = cfg["Model"]

    # Build model Config
    encoder_cfg = WSIEncoderConfig(
        tile_dim=int(model_cfg["tile_dim"]),
        roi_dim=int(model_cfg["roi_fused_dim"]),
        mil_hidden=int(model_cfg["mil_hidden"]),
        topk=int(model_cfg["topk"]),
        active_n_rois=int(model_cfg.get("active_n_rois", 0)),
        hash_bits=int(model_cfg.get("hash_bits", 64)),
        hash_chunk_size=int(model_cfg.get("hash_chunk_size", 1024)),
        hash_guide_alpha=float(model_cfg.get("hash_guide_alpha", 0.2)),
        hash_on_mag=model_cfg.get("hash_on_mag", [0,1]),
        hash_rank=int(model_cfg.get("hash_rank", 64)),
    )

    text_encoder_name = cfg.get("Text", {}).get("text_encoder_name", None)
    if args.use_clinical_text and (text_encoder_name is None or text_encoder_name == ""):
        print("Warning: --use_clinical_text is set but Text.text_encoder_name is empty; running without text branch.")
    model_text_encoder_name = text_encoder_name if args.use_clinical_text else None
    
    model = WSIDownstreamModel(
        encoder_cfg, 
        task_type=args.task, 
        num_classes=4 if args.task=="stage" else 1,
        tune_strategy=args.tune_strategy,
        text_encoder_name=model_text_encoder_name
    ).to(device)

    # Load frozen weights
    model.load_pretrained_encoder(args.checkpoint)

    # Optimizer
    if args.tune_strategy in ["finetune", "partial"]:
        encoder_params = [p for p in model.wsi_encoder.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": args.learning_rate * 0.1},
            {"params": head_params, "lr": args.learning_rate}
        ], weight_decay=0.1)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.1)

    # Datasets Path
    split_dir = os.path.dirname(data_cfg["csv_path"])
    default_train_csv = os.path.join(split_dir, "train.csv")
    default_val_csv = os.path.join(split_dir, "val.csv")
    default_test_csv = os.path.join(split_dir, "test.csv")
    train_csv = args.train_csv if args.train_csv else default_train_csv
    eval_csv = args.test_csv if args.test_csv else (args.val_csv if args.val_csv else default_val_csv)
    if args.test_csv and not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV override not found: {args.test_csv}")
    if args.train_csv and not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Train CSV override not found: {args.train_csv}")

    # Class weights for Stage task
    class_weights = None
    if args.task == "stage":
        train_df = pd.read_csv(train_csv)
        valid_labels = train_df[train_df["stage_label"] != -1]["stage_label"]
        counts = valid_labels.value_counts().to_dict()
        n_samples = len(valid_labels)
        n_classes = 4
        weights = [n_samples / (n_classes * counts.get(i, 1.0)) for i in range(n_classes)]
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        print(f"Computed Class Weights: {weights}")

    # Dataset Init
    n_rois_by_mag = data_cfg.get("n_rois_by_mag", {})
    report_col = "clinical_history" if args.use_clinical_text else data_cfg.get("report_col", "report_text")
    
    # 🔥 修复4：传入task参数，适配数据集过滤逻辑
    train_ds = DownstreamDataset(train_csv, data_cfg["feature_dir"], task=args.task, 
                                 target_mags=data_cfg["mags"], n_rois_by_mag=n_rois_by_mag, report_col=report_col)
    eval_ds = DownstreamDataset(eval_csv, data_cfg["feature_dir"], task=args.task,
                                target_mags=data_cfg["mags"], n_rois_by_mag=n_rois_by_mag, report_col=report_col)
    
    print(f"Using report column: {report_col}")
    print(f"Using clinical text branch: {args.use_clinical_text and model.use_text}")
    print(f"Training from: {train_csv} | Samples: {len(train_ds)}")
    print(f"Evaluating on: {eval_csv} | Samples: {len(eval_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_downstream)
    val_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_downstream)

    os.makedirs(args.output_dir, exist_ok=True)
    best_metric = 0.0

    # 🔥 修复5：预存完整训练集生存数据（验证指标稳定可比）
    full_train_times, full_train_events, full_train_scores = [], [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            mags_list = batch["mags"]
            features_by_mag, masks_by_mag, coords_by_mag = {}, {}, {}
            for mag in mags_list:
                features_by_mag[mag] = batch[f"feat_{mag}"].to(device)
                masks_by_mag[mag] = batch[f"mask_{mag}"].to(device)
                if batch.get(f"coords_{mag}") is not None:
                    coords_by_mag[mag] = batch[f"coords_{mag}"].to(device)
            level0_size = batch.get("level0_size").to(device) if batch.get("level0_size") is not None else None

            # Get target
            if args.task == "stage":
                targets = batch["stage_label"].to(device)
                valid_mask = targets != -1
            else:
                surv_time = batch["surv_time"].to(device)
                surv_status = batch["surv_status"].to(device)
                # 🔥 修复6：浮点数精度安全判断，避免-1.0误判
                valid_mask = (torch.abs(surv_time + 1.0) > 1e-6) & (surv_status != -1)

            if valid_mask.sum() == 0:
                continue

            optimizer.zero_grad()
            
            texts = batch.get("clinical_history") if (args.use_clinical_text and model.use_text) else None
            preds = model(features_by_mag, masks_by_mag, coords_by_mag, level0_size, texts=texts)
            
            # Loss Calculation
            if args.task == "stage":
                loss = F.cross_entropy(preds[valid_mask], targets[valid_mask], weight=class_weights)
            else:
                loss = cox_ph_loss(preds[valid_mask], surv_time[valid_mask], surv_status[valid_mask])
                # 收集完整训练集生存数据
                with torch.no_grad():
                    full_train_times.extend(surv_time[valid_mask].detach().cpu().numpy())
                    full_train_events.extend(surv_status[valid_mask].detach().cpu().numpy())
                    full_train_scores.extend(preds[valid_mask].squeeze(-1).detach().cpu().numpy())

            # NaN Loss 检查
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping batch due to NaN/Inf loss")
                optimizer.zero_grad()
                continue

            # 反向传播 + 梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # 🔥 修复7：梯度NaN/Inf检查
            has_nan_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) 
                              for p in model.parameters())
                    
            if has_nan_grad:
                print("Skipping step due to NaN/Inf gradients")
                optimizer.zero_grad()
                continue
                
            optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({"loss": loss.item(), "avg_loss": train_loss/valid_batches})

        # Validation
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        all_times, all_status = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                mags_list = batch["mags"]
                features_by_mag, masks_by_mag, coords_by_mag = {}, {}, {}
                for mag in mags_list:
                    features_by_mag[mag] = batch[f"feat_{mag}"].to(device)
                    masks_by_mag[mag] = batch[f"mask_{mag}"].to(device)
                    if batch.get(f"coords_{mag}") is not None:
                        coords_by_mag[mag] = batch[f"coords_{mag}"].to(device)
                level0_size = batch.get("level0_size").to(device) if batch.get("level0_size") is not None else None

                if args.task == "stage":
                    targets = batch["stage_label"].to(device)
                    valid_mask = targets != -1
                else:
                    surv_time = batch["surv_time"].to(device)
                    surv_status = batch["surv_status"].to(device)
                    valid_mask = (torch.abs(surv_time + 1.0) > 1e-6) & (surv_status != -1)

                if valid_mask.sum() == 0:
                    continue
                
                texts = batch.get("clinical_history") if (args.use_clinical_text and model.use_text) else None
                preds = model(features_by_mag, masks_by_mag, coords_by_mag, level0_size, texts=texts)
                    
                if args.task == "stage":
                    probs = torch.softmax(preds[valid_mask], dim=1).detach().cpu()
                    preds_class = torch.argmax(probs, dim=1)
                    all_preds.extend(preds_class.numpy())
                    all_probs.extend(probs.numpy())
                    all_targets.extend(targets[valid_mask].cpu().numpy())
                else:
                    risk = preds[valid_mask].squeeze(-1).detach().cpu()
                    all_preds.extend(risk.numpy())
                    all_times.extend(surv_time[valid_mask].cpu().numpy())
                    all_status.extend(surv_status[valid_mask].cpu().numpy())

        # Metrics Calculation
        print(f"\n--- Epoch {epoch+1} Results ---")
        current_metric = 0.0
        avg_train_loss = train_loss / valid_batches if valid_batches > 0 else 0.0
        log_msg = f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}"
        
        if args.task == "stage" and len(all_preds) > 0:
            all_probs_arr = np.array(all_probs)
            acc = accuracy_score(all_targets, all_preds)
            mac_f1 = f1_score(all_targets, all_preds, average="macro")
            cm = confusion_matrix(all_targets, all_preds)
            
            try:
                roc_auc = roc_auc_score(all_targets, all_probs_arr, multi_class="ovo", average="macro")
            except Exception as e:
                roc_auc = 0.0
                print(f"ROC AUC failed: {e}")
            
            print(f"Accuracy: {acc:.4f} | Macro F1: {mac_f1:.4f} | ROC AUC: {roc_auc:.4f}")
            print("Confusion Matrix:\n", cm)
            log_msg += f" - Val Acc: {acc:.4f} - Macro F1: {mac_f1:.4f} - ROC AUC: {roc_auc:.4f}"
            current_metric = acc

        elif args.task == "survival" and len(all_preds) > 0:
            # 🔥 修复8：使用完整训练集计算IPCW C-index
            metrics = compute_survival_metrics(
                full_train_times, full_train_events, full_train_scores,
                np.array(all_times), np.array(all_status), np.array(all_preds)
            )
            c_index = metrics["c_index"]
            print(f"C-Index (IPCW): {c_index:.4f}")
            log_msg += f" - Val C-Idx: {c_index:.4f}"
            current_metric = c_index
        
        # Save Best Model
        if current_metric > best_metric and not math.isnan(avg_train_loss):
            best_metric = current_metric
            save_path = os.path.join(args.output_dir, f"best_{args.task}_model.pth")
            torch.save(model.state_dict(), save_path)
            if args.task == "stage" and len(all_preds) > 0:
                np.save(os.path.join(args.output_dir, f"best_stage_cm.npy"), cm)
            print(f">>> New best model saved (metric: {best_metric:.4f})")
            log_msg += " [NEW BEST]"

        # NaN Loss 终止训练
        if math.isnan(avg_train_loss):
            print("NaN loss detected! Stopping training.")
            break

        # Write Log
        log_file_path = os.path.join(args.output_dir, f"training_log_{args.task}.txt")
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

if __name__ == "__main__":
    main()