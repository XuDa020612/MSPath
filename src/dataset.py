from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairSample:
    case_id: str
    wsi_path: str
    report_text: str
    clinical_history: str = ""

class WSIPairCSV(Dataset):
    """Minimal dataset reading (case_id, wsi_path, report_text, [clinical_history]) from a CSV."""

    def __init__(self, csv_path: str, report_col: str = "report_text"):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.report_col = report_col
        required = {"case_id", "wsi_path", report_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> PairSample:
        row = self.df.iloc[idx]
        clinical_text = str(row["clinical_history"]) if "clinical_history" in row else ""
        return PairSample(
            case_id=str(row["case_id"]),
            wsi_path=str(row["wsi_path"]),
            report_text=str(row[self.report_col]),
            clinical_history=clinical_text,
        )


class ROIFeatureDataset(Dataset):
    """Loads pre-extracted WSI tile features.

    Expected pack content (new format):
        - 'roi_feat': FloatTensor [N, D]
        - 'roi_mag': IntTensor [N]   (actual magnification per token)
        - 'coords_level0': IntTensor [N, 2] (optional)

    Legacy packs with shape [N, 3, D] are automatically flattened.
    """

    def __init__(
        self,
        csv_path: str,
        feature_dir: str,
        report_col: str = "report_text",
        n_rois: int = -1,
        target_mags: List[int] = None,
        n_rois_by_mag: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.pairs = WSIPairCSV(csv_path, report_col=report_col)
        self.feature_dir = feature_dir
        self.n_rois = n_rois
        self.target_mags = target_mags if target_mags is not None else [5, 10, 20]
        self.n_rois_by_mag = n_rois_by_mag or {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.pairs[idx]
        feat_path = os.path.join(self.feature_dir, f"{sample.case_id}.pt")
        
        # --- Missing File Handling ---
        if not os.path.exists(feat_path):
            print(f"Warning: Feature file missing for {sample.case_id} at {feat_path}. Skipping.")
            # Recursive retry with random other index
            # This is a common trick in PyTorch Datasets
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_idx)
            
        try:
            pack = torch.load(feat_path, map_location="cpu")
        except Exception as e:
            print(f"Warning: Corrupted feature file for {sample.case_id} at {feat_path}: {e}. Skipping.")
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_idx)
        # -----------------------------
        
        roi_feat = pack["roi_feat"].float()
        roi_mag = pack.get("roi_mag")
        coords = pack.get("coords_level0")
        mags = pack.get("mags", None)
        
        if roi_feat.dim() == 3:  # legacy [N,3,D]
            n, m, d = roi_feat.shape
            roi_feat = roi_feat.reshape(n * m, d)
            if roi_mag is None:
                mag_values = mags if mags is not None else list(range(m))
                mag_tensor = torch.tensor(mag_values, dtype=torch.int32)
                roi_mag = mag_tensor.unsqueeze(0).repeat(n, 1).reshape(-1)
            if coords is not None:
                coords = coords.to(torch.int32)
                coords = coords.unsqueeze(1).repeat(1, m, 1).reshape(n * m, 2)
        
        if roi_mag is None:
            raise ValueError(f"Feature pack {feat_path} missing 'roi_mag'")
        if coords is not None:
            coords = coords.to(torch.int32)
        
        wsi_path = pack.get("wsi_path", sample.wsi_path)
        level0_size = pack.get("level0_size", None)
        
        # Split by magnification
        out_data = {
            "case_id": sample.case_id,
            "wsi_path": wsi_path,
            "level0_size": level0_size,
            "report_text": sample.report_text,
            "clinical_history": sample.clinical_history,
            "mags": self.target_mags
        }
        
        for mag in self.target_mags:
            mask = (roi_mag == mag)
            feat_mag = roi_feat[mask]
            coords_mag = coords[mask] if coords is not None else None
            
            # Sampling per mag if cap is set
            cap = self.n_rois_by_mag.get(mag, self.n_rois)
            if cap > 0 and feat_mag.shape[0] > cap:
                perm = torch.randperm(feat_mag.shape[0])[:cap]
                feat_mag = feat_mag[perm]
                if coords_mag is not None:
                    coords_mag = coords_mag[perm]
            
            out_data[f"feat_{mag}"] = feat_mag
            out_data[f"coords_{mag}"] = coords_mag
            
        return out_data


def collate_pairs(batch: List[Dict]) -> Dict:
    case_ids = [b["case_id"] for b in batch]
    report_text = [b["report_text"] for b in batch]
    wsi_paths = [b.get("wsi_path", "") for b in batch]
    mags = batch[0]["mags"]
    clinical_history = [b.get("clinical_history", "") for b in batch]
    
    level0_sizes = []
    for b in batch:
        size = b.get("level0_size")
        if size is None:
            level0_sizes.append((1.0, 1.0))
        else:
            level0_sizes.append((float(size[0]), float(size[1])))
    level0_size = torch.tensor(level0_sizes, dtype=torch.float32)

    out_batch = {
        "case_id": case_ids,
        "wsi_path": wsi_paths,
        "report_text": report_text,
        "clinical_history": clinical_history,
        "level0_size": level0_size,
        "mags": mags
    }
    
    bsz = len(batch)
    
    for mag in mags:
        feats = [b[f"feat_{mag}"] for b in batch]
        lengths = [f.shape[0] for f in feats]
        max_n = max(lengths) if lengths else 0
        d = feats[0].shape[-1] if feats else 768
        
        if max_n == 0:
            # Handle empty case
            out_batch[f"feat_{mag}"] = torch.zeros((bsz, 0, d), dtype=torch.float32)
            out_batch[f"mask_{mag}"] = torch.zeros((bsz, 0), dtype=torch.bool)
            out_batch[f"coords_{mag}"] = None
            continue
            
        batch_feat = torch.zeros((bsz, max_n, d), dtype=torch.float32)
        batch_mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        
        coords_list = [b[f"coords_{mag}"] for b in batch]
        coords_enabled = any(c is not None for c in coords_list)
        batch_coords = torch.zeros((bsz, max_n, 2), dtype=torch.int32) if coords_enabled else None
        
        for i in range(bsz):
            n = lengths[i]
            if n > 0:
                batch_feat[i, :n] = feats[i]
                batch_mask[i, :n] = True
                if batch_coords is not None and coords_list[i] is not None:
                    batch_coords[i, :n] = coords_list[i]
                    
        out_batch[f"feat_{mag}"] = batch_feat
        out_batch[f"mask_{mag}"] = batch_mask
        out_batch[f"coords_{mag}"] = batch_coords
        
    return out_batch
