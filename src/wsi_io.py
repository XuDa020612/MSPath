import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class WSILevels:
    mag_to_level: Dict[int, int]
    downsample: Dict[int, float]
    level0_size: Tuple[int, int]


def _get_float(props: Dict[str, str], key: str):
    v = props.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def estimate_mag0(slide_props: Dict[str, str]) -> Optional[float]:
    # Best-effort: prefer objective power.
    mag0 = _get_float(slide_props, "openslide.objective-power")
    if mag0 is not None:
        return mag0

    # Fallback: infer from MPP (assumes 0.5 um/px ~ 20x; 0.25 ~ 40x).
    mpp_x = _get_float(slide_props, "openslide.mpp-x")
    if mpp_x is None:
        return None

    # heuristic: mag ≈ 10 / mpp (so 0.5 -> 20x, 0.25 -> 40x)
    return 10.0 / mpp_x


def choose_levels_for_mags(slide, target_mags: List[int]) -> WSILevels:
    """Pick the closest OpenSlide level for each requested magnification.

    Assumes level0 is at magnification mag0, and mag[level] ≈ mag0 / downsample[level].
    """
    props = dict(getattr(slide, "properties", {}))
    mag0 = estimate_mag0(props)
    if mag0 is None:
        raise RuntimeError(
            "Cannot determine level0 magnification (missing openslide.objective-power and openslide.mpp-x)."
        )

    downsample = {level: float(slide.level_downsamples[level]) for level in range(slide.level_count)}
    level0_w, level0_h = slide.level_dimensions[0]

    mag_to_level: Dict[int, int] = {}
    for m in target_mags:
        best_level = None
        best_err = None
        for level, ds in downsample.items():
            mag_level = mag0 / ds
            err = abs(mag_level - float(m))
            if best_err is None or err < best_err:
                best_err = err
                best_level = level
        if best_level is None:
            raise RuntimeError(f"Failed to select level for magnification {m}")
        mag_to_level[int(m)] = int(best_level)

    return WSILevels(mag_to_level=mag_to_level, downsample=downsample, level0_size=(level0_w, level0_h))


def build_tissue_mask_from_thumbnail(rgb: np.ndarray) -> np.ndarray:
    """Build a tissue mask from a thumbnail RGB image.

    Default behavior keeps backward compatibility: a simple gray-threshold mask
    with a tiny majority filter. For higher quality on many slides, prefer
    `build_tissue_mask_from_thumbnail_hsv`.
    """
    return build_tissue_mask_from_thumbnail_gray(rgb)


def _majority_filter(mask: np.ndarray, k: int = 3, min_sum: int = 5) -> np.ndarray:
    """Small dependency-free majority filter for boolean masks.
    Optimized using scipy.ndimage.convolve for speed.
    """
    try:
        from scipy.ndimage import convolve
        k = int(k)
        if k <= 1:
            return mask.astype(bool)
        
        kernel = np.ones((k, k), dtype=int)
        m = mask.astype(int)
        # constant mode with 0 padding is equivalent to the original manual padding
        conv = convolve(m, kernel, mode='constant', cval=0)
        return conv >= min_sum
    except ImportError:
        # Fallback to slow implementation if scipy is missing
        k = int(k)
        if k <= 1:
            return mask.astype(bool)
        pad = k // 2
        m = mask.astype(np.uint8)
        m_pad = np.pad(m, ((pad, pad), (pad, pad)), mode="constant")
        out = np.zeros_like(m)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                window = m_pad[y : y + k, x : x + k]
                out[y, x] = 1 if int(window.sum()) >= int(min_sum) else 0
        return out.astype(bool)


def build_tissue_mask_from_thumbnail_gray(rgb: np.ndarray, gray_thresh: float = 0.9, k: int = 3, min_sum: int = 5) -> np.ndarray:
    """Simple tissue mask: gray threshold + majority filter."""
    img = rgb.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    mask = gray < float(gray_thresh)
    return _majority_filter(mask, k=k, min_sum=min_sum)


def build_tissue_mask_from_thumbnail_hsv(
    rgb: np.ndarray,
    sat_thresh: float = 0.05,
    val_thresh: float = 0.95,
    gray_thresh: float = 0.92,
    k: int = 3,
    min_sum: int = 5,
) -> np.ndarray:
    """More robust tissue mask: combine saturation/value/gray cues.

    This mimics the spirit of CLAM's HSV-based segmentation (without cv2):
    background tends to have low saturation and high value.
    """
    img = rgb.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = np.zeros_like(maxc)
    nz = maxc > 1e-6
    s[nz] = delta[nz] / maxc[nz]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # Tissue is often darker OR saturated; background is bright and desaturated.
    tissue = (s > float(sat_thresh)) | (gray < float(gray_thresh))
    # Remove very bright pixels even if slightly saturated (glare/white background)
    tissue = tissue & (v < float(val_thresh))

    return _majority_filter(tissue, k=k, min_sum=min_sum)


def build_tissue_mask_from_thumbnail_otsu(
    rgb: np.ndarray,
    mthresh: int = 7,
    close_ksize: int = 4,
    min_sum: int = 0, # Unused, kept for backwards compatibility in args
    filter_k: int = 0
) -> np.ndarray:
    """Build a tissue mask utilizing the exact Otsu method from CLAM.
    
    Reproduces CLAM WholeSlideImage.segmentTissue logic:
    HSV conversion -> Saturation channel -> Median Blur -> Otsu Threshold -> Closing.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("OpenCV is required for Otsu tissue segmentation") from e

    img = rgb.astype(np.uint8)
    
    # 1. Convert to HSV and extract median-blurred Saturation channel
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    
    # 2. Otsu thresholding (CLAM use_otsu=True branch)
    _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    
    # 3. Morphological closing 
    if close_ksize > 0:
        kernel = np.ones((close_ksize, close_ksize), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        
    return img_otsu > 0


def sample_roi_centers_level0(
    tissue_mask: np.ndarray,
    thumb_to_level0_scale: Tuple[float, float],
    n_rois: int,
    rng: np.random.Generator,
    margin_level0: int,
    level0_size: Tuple[int, int],
) -> np.ndarray:
    """Sample ROI centers (x0,y0) in level0 pixel coordinates."""
    ys, xs = np.where(tissue_mask)
    if len(xs) == 0:
        raise RuntimeError("No tissue pixels detected in tissue mask.")

    level0_w, level0_h = level0_size
    sx, sy = thumb_to_level0_scale

    centers = []
    tries = 0
    max_tries = n_rois * 50

    while len(centers) < n_rois and tries < max_tries:
        tries += 1
        idx = int(rng.integers(0, len(xs)))
        x_t, y_t = int(xs[idx]), int(ys[idx])
        x0 = int(round(x_t * sx))
        y0 = int(round(y_t * sy))

        if x0 < margin_level0 or x0 > level0_w - margin_level0:
            continue
        if y0 < margin_level0 or y0 > level0_h - margin_level0:
            continue
        centers.append((x0, y0))

    if len(centers) < n_rois:
        raise RuntimeError(f"Failed to sample enough ROI centers: {len(centers)}/{n_rois}")

    return np.array(centers, dtype=np.int32)


def sample_roi_centers_level0_unique(
    tissue_mask: np.ndarray,
    thumb_to_level0_scale: Tuple[float, float],
    n_rois: int,
    rng: np.random.Generator,
    margin_level0: int,
    level0_size: Tuple[int, int],
    max_candidates_factor: int = 50,
) -> np.ndarray:
    """Sample unique ROI centers (x0,y0) in level0 pixel coordinates.

    Strategy:
    - Prefer sampling without replacement from tissue pixels on thumbnail.
    - Deduplicate after mapping to level0 coords.
    - Fallback to random draws with replacement if still insufficient.
    """
    ys, xs = np.where(tissue_mask)
    if len(xs) == 0:
        raise RuntimeError("No tissue pixels detected in tissue mask.")

    level0_w, level0_h = level0_size
    sx, sy = thumb_to_level0_scale

    centers: List[Tuple[int, int]] = []
    seen = set()

    # 1) Without replacement pass.
    idxs = np.arange(len(xs))
    rng.shuffle(idxs)
    for idx in idxs:
        x_t, y_t = int(xs[idx]), int(ys[idx])
        x0 = int(round(x_t * sx))
        y0 = int(round(y_t * sy))

        if x0 < margin_level0 or x0 > level0_w - margin_level0:
            continue
        if y0 < margin_level0 or y0 > level0_h - margin_level0:
            continue

        key = (x0, y0)
        if key in seen:
            continue
        seen.add(key)
        centers.append(key)
        if len(centers) >= n_rois:
            return np.array(centers, dtype=np.int32)

    # 2) Fallback: random draws with replacement (still dedup).
    tries = 0
    max_tries = max(1, n_rois * max_candidates_factor)
    while len(centers) < n_rois and tries < max_tries:
        tries += 1
        idx = int(rng.integers(0, len(xs)))
        x_t, y_t = int(xs[idx]), int(ys[idx])
        x0 = int(round(x_t * sx))
        y0 = int(round(y_t * sy))

        if x0 < margin_level0 or x0 > level0_w - margin_level0:
            continue
        if y0 < margin_level0 or y0 > level0_h - margin_level0:
            continue

        key = (x0, y0)
        if key in seen:
            continue
        seen.add(key)
        centers.append(key)

    if len(centers) < n_rois:
        raise RuntimeError(f"Failed to sample enough unique ROI centers: {len(centers)}/{n_rois}")

    return np.array(centers, dtype=np.int32)


def compute_read_size_for_same_fov(
    out_patch_size: int,
    base_level: int,
    target_level: int,
    downsample: Dict[int, float],
) -> int:
    """Compute pixel size at target_level so that FOV matches out_patch_size at base_level."""
    ds_base = float(downsample[base_level])
    ds_tgt = float(downsample[target_level])
    # same physical FOV: px_tgt * ds_tgt == out_patch_size * ds_base
    px = out_patch_size * ds_base / ds_tgt
    return int(max(1, round(px)))


def center_to_top_left_level0(x0: int, y0: int, read_size_level: int, level: int, downsample: Dict[int, float]) -> Tuple[int, int]:
    """Convert center (level0 coords) to top-left (level0 coords) for read_region."""
    ds = float(downsample[level])
    half = int(round((read_size_level * ds) / 2.0))
    return x0 - half, y0 - half
