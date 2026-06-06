from __future__ import annotations

import argparse
import math
import os
import sys
import yaml
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import pandas as pd

# Make imports robust regardless of cwd.
WSI_REPORT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WSI_REPORT_ROOT not in sys.path:
    sys.path.insert(0, WSI_REPORT_ROOT)

from src.runtime_paths import add_repo_root_to_sys_path
add_repo_root_to_sys_path()

from models.ctran import ctranspath
from src.wsi_io import (
    build_tissue_mask_from_thumbnail_gray,
    build_tissue_mask_from_thumbnail_hsv,
    build_tissue_mask_from_thumbnail_otsu,
    choose_levels_for_mags,
    estimate_mag0,
)

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python is required for WSI ROI extraction") from e


class WSIPatchDataset(Dataset):
    def __init__(self, wsi_path, coords, levels, patch_size, out_patch_size, transform):
        self.wsi_path = wsi_path
        self.coords = coords  # List of (x, y, level, mag)
        self.patch_size = patch_size
        self.out_patch_size = out_patch_size
        self.transform = transform
        # Do not initialize slide here to avoid pickling issues

    def __getitem__(self, idx):
        # Initialize slide in the worker process
        if not hasattr(self, '_slide'):
            self._slide = openslide.OpenSlide(self.wsi_path)
            
        x, y, level, mag = self.coords[idx]
        
        try:
            patch = self._slide.read_region((x, y), level, (self.patch_size, self.patch_size))
            patch = patch.convert("RGB")
            
            if self.patch_size != self.out_patch_size:
                patch = patch.resize((self.out_patch_size, self.out_patch_size), resample=Image.BILINEAR)
            
            if self.transform:
                patch = self.transform(patch)
                
            return patch, torch.tensor([x, y]), torch.tensor(mag)
        except Exception as e:
            print(f"Error reading region {x},{y} at level {level}: {e}")
            # Return a dummy tensor in case of error
            return torch.zeros((3, self.out_patch_size, self.out_patch_size)), torch.tensor([x, y]), torch.tensor(mag)

    def __len__(self):
        return len(self.coords)


def level0_patch_size(tile_px: int, downsample: float) -> int:
    return max(1, int(round(tile_px * float(downsample))))

def tile_iter(level_w: int, level_h: int, tile_px: int, stride_px: int) -> Iterator[Tuple[int, int]]:
    stride_px = max(1, int(stride_px))
    for y in range(0, max(1, level_h - tile_px + 1), stride_px):
        if y + tile_px > level_h:
            break
        for x in range(0, max(1, level_w - tile_px + 1), stride_px):
            if x + tile_px > level_w:
                break
            yield x, y

def compute_tissue_fraction(
    mask: np.ndarray,
    level0_x0: int,
    level0_y0: int,
    level0_w: int,
    level0_h: int,
    patch_w0: int,
    patch_h0: int,
    thumb_w: int,
    thumb_h: int,
) -> float:
    if mask.size == 0:
        return 0.0
    sx = thumb_w / float(level0_w)
    sy = thumb_h / float(level0_h)
    x0_thumb = int(math.floor(level0_x0 * sx))
    x1_thumb = int(math.ceil((level0_x0 + patch_w0) * sx))
    y0_thumb = int(math.floor(level0_y0 * sy))
    y1_thumb = int(math.ceil((level0_y0 + patch_h0) * sy))
    x0_thumb = max(0, min(x0_thumb, thumb_w - 1))
    y0_thumb = max(0, min(y0_thumb, thumb_h - 1))
    x1_thumb = max(x0_thumb + 1, min(x1_thumb, thumb_w))
    y1_thumb = max(y0_thumb + 1, min(y1_thumb, thumb_h))
    
    if x1_thumb <= x0_thumb or y1_thumb <= y0_thumb:
        return 0.0
        
    patch = mask[y0_thumb:y1_thumb, x0_thumb:x1_thumb]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config for extraction params")
    parser.add_argument("--csv", default=None, help="CSV with columns: case_id,wsi_path,report_text")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--mags", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--use_levels",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="If set, ignore --mags and extract using OpenSlide level ids (e.g. 0 1 2). Tokens will be labeled by level id.",
    )
    parser.add_argument("--base_mag", type=int, default=20)
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size (pixels) at each requested magnification")
    parser.add_argument("--tile_stride", type=int, default=256, help="Stride for tiling (pixels). Use <tile_size for overlap)")
    parser.add_argument(
        "--tile_stride_by_level",
        type=str,
        default=None,
        help="Optional per-level stride overrides, e.g. '0:256,1:256,2:128'",
    )
    parser.add_argument("--out_patch_size", type=int, default=224)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <case_id>.pt in out_dir instead of skipping.",
    )
    parser.add_argument(
        "--tissue_threshold",
        type=float,
        default=0.3,
        help="Minimum tissue ratio (0-1) required to keep a tile",
    )
    parser.add_argument(
        "--tissue_threshold_by_level",
        type=str,
        default=None,
        help="Optional per-level tissue thresholds, e.g. '0:0.4,1:0.3,2:0.2'",
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        default="gray",
        choices=["gray", "hsv", "otsu"],
        help="Tissue mask method on thumbnail. 'hsv' is usually more robust. 'otsu' learns from CLAM.",
    )
    parser.add_argument("--mask_gray_thresh", type=float, default=0.9)
    parser.add_argument("--mask_sat_thresh", type=float, default=0.05)
    parser.add_argument("--mask_val_thresh", type=float, default=0.95)
    parser.add_argument("--mask_morph_k", type=int, default=3)
    parser.add_argument("--mask_close_ksize", type=int, default=7)
    parser.add_argument("--mask_morph_min_sum", type=int, default=5)
    parser.add_argument("--model_weight_dir", default="../../model_weight")
    parser.add_argument("--encode_batch", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg:
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    def parse_kv_map(raw: Optional[str], cast_type=float):
        if not raw:
            return {}
        out = {}
        for item in str(raw).split(","):
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            out[int(k.strip())] = cast_type(v.strip())
        return out

    stride_by_level = parse_kv_map(args.tile_stride_by_level, cast_type=int)
    tissue_by_level = parse_kv_map(args.tissue_threshold_by_level, cast_type=float)

    if args.csv is None or args.out_dir is None:
        parser.error("--csv and --out_dir are required (either via CLI or config)")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    model = ctranspath()
    model.head = nn.Identity()
    weight_path = os.path.join(args.model_weight_dir, "ctranspath.pth")
    if not os.path.exists(weight_path):
        # Fallback or error
        print(f"Weight file not found at {weight_path}, trying absolute path...")
        # Try to find it in the workspace if possible, but for now just warn
    
    try:
        td = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(td["model"], strict=True)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
        
    model.eval().to(device)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    tr = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    df = pd.read_csv(args.csv)
    print(f"Processing {len(df)} slides...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = str(row["case_id"])
        wsi_path = str(row["wsi_path"])
        out_path = os.path.join(args.out_dir, f"{case_id}.pt")
        
        if os.path.exists(out_path) and not args.overwrite:
            continue

        try:
            slide = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"Error opening slide {wsi_path}: {e}")
            continue

        try:
            levels = choose_levels_for_mags(slide, target_mags=args.mags)
        except Exception as e:
            if args.use_levels is not None and len(args.use_levels) > 0:
                from src.wsi_io import WSILevels
                downsample = {lvl: float(slide.level_downsamples[lvl]) for lvl in range(slide.level_count)}
                lw, lh = slide.level_dimensions[0]
                mock_mag2lvl = {m: 0 for m in args.mags}
                mock_mag2lvl[args.base_mag] = 0
                levels = WSILevels(mag_to_level=mock_mag2lvl, downsample=downsample, level0_size=(lw, lh))
            else:
                print(f"Error choosing levels for {case_id}: {e}")
                continue

        use_levels = args.use_levels is not None and len(args.use_levels) > 0
        if use_levels:
            # Validate requested levels exist.
            req_levels = [int(x) for x in args.use_levels]
            bad = [lv for lv in req_levels if lv < 0 or lv >= int(slide.level_count)]
            if bad:
                print(f"[warning] {case_id}: requested levels out of range for slide(level_count={slide.level_count}): {bad}. Skipping.")
                continue
            targets = sorted(set(req_levels))
            scale_type = "level"
        else:
            targets = [int(m) for m in args.mags]
            scale_type = "mag"

        # Warn if multiple requested magnifications map to the same OpenSlide level (mag mode only).
        if not use_levels:
            lvl_to_mags = defaultdict(list)
            for m, lvl in levels.mag_to_level.items():
                lvl_to_mags[int(lvl)].append(int(m))
            dup = {lvl: ms for lvl, ms in lvl_to_mags.items() if len(ms) > 1}
            if dup:
                mag0 = estimate_mag0(dict(getattr(slide, "properties", {})))
                avail = []
                if mag0 is not None:
                    for lv in range(slide.level_count):
                        ds = float(slide.level_downsamples[lv])
                        avail.append((lv, mag0 / ds, ds))
                msg = f"[warning] {case_id}: requested mags {list(args.mags)} map to duplicate OpenSlide levels: {dup}. "
                if avail:
                    msg += "Available level mags (approx): " + ", ".join(
                        [f"level{lv}:{m:.2f}x(ds={ds:g})" for lv, m, ds in avail]
                    )
                print(msg)

        base_level = levels.mag_to_level[int(args.base_mag)]

        # Generate tissue mask
        try:
            thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
            thumb_np = np.array(thumb)
            if args.mask_method == "hsv":
                mask = build_tissue_mask_from_thumbnail_hsv(
                    thumb_np,
                    sat_thresh=args.mask_sat_thresh,
                    val_thresh=args.mask_val_thresh,
                    gray_thresh=max(args.mask_gray_thresh, 0.85),
                    k=args.mask_morph_k,
                    min_sum=args.mask_morph_min_sum,
                )
            elif args.mask_method == "otsu":
                mask = build_tissue_mask_from_thumbnail_otsu(
                    thumb_np,
                    mthresh=args.mask_morph_k,
                    close_ksize=args.mask_close_ksize,
                )
            else:
                mask = build_tissue_mask_from_thumbnail_gray(
                    thumb_np,
                    gray_thresh=args.mask_gray_thresh,
                    k=args.mask_morph_k,
                    min_sum=args.mask_morph_min_sum,
                )
        except Exception as e:
            print(f"Error generating mask for {case_id}: {e}")
            continue

        level0_w, level0_h = levels.level0_size
        thumb_w, thumb_h = thumb.size

        # Collect all valid coordinates first
        valid_coords = []
        counts_by_mag = defaultdict(int)
        read_sizes = {}

        for t in targets:
            if use_levels:
                lvl = int(t)
                label = int(t)  # label by level id
                down = float(slide.level_downsamples[lvl])
            else:
                mag = int(t)
                lvl = int(levels.mag_to_level[mag])
                label = int(mag)  # label by magnification
                down = float(levels.downsample[lvl])

            read_sizes[label] = args.tile_size

            level_w, level_h = slide.level_dimensions[lvl]
            tile_px = int(args.tile_size)
            stride_px = int(stride_by_level.get(lvl, args.tile_stride))
            patch_w0 = level0_patch_size(tile_px, down)
            patch_h0 = patch_w0

            tissue_thr = float(tissue_by_level.get(lvl, args.tissue_threshold))

            for x_lvl, y_lvl in tile_iter(level_w, level_h, tile_px, stride_px):
                x0_level0 = int(round(x_lvl * down))
                y0_level0 = int(round(y_lvl * down))
                
                if x0_level0 + patch_w0 > level0_w or y0_level0 + patch_h0 > level0_h:
                    continue

                frac = compute_tissue_fraction(
                    mask, x0_level0, y0_level0, level0_w, level0_h,
                    patch_w0, patch_h0, thumb_w, thumb_h
                )
                
                if frac >= tissue_thr:
                    # Store level0 coordinates for reading
                    # Note: read_region expects level 0 coordinates for x,y
                    # But we need to pass the coordinates appropriate for the level if we were using read_region(..., level, ...)
                    # OpenSlide read_region(location, level, size)
                    # location: (x, y) tuple giving the top left pixel in the level 0 reference frame.
                    # So we should pass x0_level0, y0_level0.
                    valid_coords.append((x0_level0, y0_level0, lvl, label))
                    counts_by_mag[label] += 1

        if not valid_coords:
            print(f"[warning] {case_id} had no valid tiles; skipping")
            continue

        # Create dataset and dataloader
        dataset = WSIPatchDataset(
            wsi_path, 
            valid_coords, 
            levels, 
            args.tile_size, 
            args.out_patch_size, 
            tr
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.encode_batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        all_feats = []
        all_coords = []
        all_mags = []

        # Inference loop
        with torch.no_grad():
            for batch_imgs, batch_coords, batch_mags in dataloader:
                batch_imgs = batch_imgs.to(device)
                feats = model(batch_imgs).detach().cpu()
                
                all_feats.append(feats)
                
                # Calculate center coordinates for storage
                # batch_coords is [B, 2] (x0, y0)
                # We need to convert to center coordinates based on mag
                for i in range(len(batch_mags)):
                    label = int(batch_mags[i])
                    if use_levels:
                        lvl = int(label)
                        down = float(slide.level_downsamples[lvl])
                    else:
                        mag = int(label)
                        lvl = int(levels.mag_to_level[mag])
                        down = float(levels.downsample[lvl])
                    patch_w0 = level0_patch_size(args.tile_size, down)
                    
                    cx = int(batch_coords[i, 0]) + patch_w0 // 2
                    cy = int(batch_coords[i, 1]) + patch_w0 // 2
                    all_coords.append((cx, cy))
                    all_mags.append(label)

        if all_feats:
            roi_feat = torch.cat(all_feats, dim=0)
            coords_tensor = torch.tensor(all_coords, dtype=torch.int32)
            mag_tensor = torch.tensor(all_mags, dtype=torch.int32)

            pack = {
                "roi_feat": roi_feat,
                "roi_mag": mag_tensor,
                "coords_level0": coords_tensor,
                "mags": list(targets),
                "scale_type": scale_type,
                "counts_by_mag": {int(k): int(v) for k, v in counts_by_mag.items()},
                "wsi_path": wsi_path,
                "base_mag": int(args.base_mag),
                "base_level": int(base_level),
                "mag_to_level": dict(levels.mag_to_level),
                "level_downsample": dict(levels.downsample),
                "level0_size": tuple(levels.level0_size),
                "out_patch_size": int(args.out_patch_size),
                "tile_size": int(args.tile_size),
                "read_sizes": read_sizes,
                "tissue_threshold": float(args.tissue_threshold),
            }
            torch.save(pack, out_path)
        else:
            print(f"No features extracted for {case_id}")

if __name__ == "__main__":
    main()
