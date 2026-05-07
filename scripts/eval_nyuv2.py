"""
Evaluate depth estimation and surface normal estimation on NYUv2 val set.

Usage:
    # Quick test with 5 samples
    python scripts/eval_nyuv2.py --model /path/to/model --data data/nyuv2 --max_samples 5

    # Full evaluation (654 samples)
    python scripts/eval_nyuv2.py --model /path/to/model --data data/nyuv2

    # Depth only
    python scripts/eval_nyuv2.py --model /path/to/model --data data/nyuv2 --task depth

    # Normals only
    python scripts/eval_nyuv2.py --model /path/to/model --data data/nyuv2 --task normals
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Metrics
# ======================================================================

def depth_metrics(pred: np.ndarray, gt: np.ndarray,
                  min_depth: float = 1e-3, max_depth: float = 10.0
                  ) -> dict[str, float]:
    """Standard monocular depth metrics on valid pixels.

    Returns delta1, delta2, delta3, abs_rel, sq_rel, rmse, rmse_log.
    """
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 10:
        return {}

    p, g = pred[valid], gt[valid]

    thresh = np.maximum(p / g, g / p)
    delta1 = float((thresh < 1.25).mean())
    delta2 = float((thresh < 1.25 ** 2).mean())
    delta3 = float((thresh < 1.25 ** 3).mean())

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))

    return dict(delta1=delta1, delta2=delta2, delta3=delta3,
                abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log)


def depth_metrics_scale_invariant(pred: np.ndarray, gt: np.ndarray,
                                  min_depth: float = 1e-3,
                                  max_depth: float = 10.0
                                  ) -> dict[str, float]:
    """Compute metrics after optimal median scaling (common for zero-shot)."""
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 10:
        return {}

    pred_median = float(np.median(pred[valid]))
    if pred_median < 1e-6:
        return {}
    scale = float(np.median(gt[valid]) / pred_median)
    pred_scaled = pred * scale
    metrics = depth_metrics(pred_scaled, gt, min_depth, max_depth)
    metrics["scale_factor"] = scale
    return metrics


def depth_metrics_affine_invariant(pred: np.ndarray, gt: np.ndarray,
                                   min_depth: float = 1e-3,
                                   max_depth: float = 10.0
                                   ) -> dict[str, float]:
    """Least-squares affine alignment: pred_aligned = a * pred + b.

    This is the fairest evaluation for zero-shot models that don't know
    the target depth scale or offset. It measures the model's ability
    to recover the *structure* (relative ordering and proportions) of depth.
    """
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred)
    if valid.sum() < 10:
        return {}

    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)

    A = np.stack([p, np.ones_like(p)], axis=-1)
    result = np.linalg.lstsq(A, g, rcond=None)
    a, b = result[0]

    pred_aligned = np.clip(pred * a + b, min_depth, max_depth)
    metrics = depth_metrics(pred_aligned, gt, min_depth, max_depth)
    metrics["affine_a"] = float(a)
    metrics["affine_b"] = float(b)
    return metrics


def _to_hwc_unit(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if n.shape[0] == 3 and n.ndim == 3:
        n = np.transpose(n, (1, 2, 0))
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-8, None), (norm.squeeze(-1) > 0.5)


def _angular_error(pred: np.ndarray, gt: np.ndarray,
                   valid: np.ndarray) -> np.ndarray:
    cos_sim = np.sum(pred * gt, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_sim))
    return angles[valid]


def normal_metrics(pred: np.ndarray, gt: np.ndarray,
                   valid_mask: np.ndarray | None = None,
                   axis_perm: tuple[int, ...] | None = None,
                   axis_signs: tuple[int, int, int] | None = None,
                   ) -> dict[str, float]:
    """Surface normal evaluation metrics.

    Args:
        pred: (H, W, 3) or (3, H, W) predicted unit normals
        gt:   (H, W, 3) or (3, H, W) ground truth unit normals
        valid_mask: optional (H, W) bool mask for valid pixels
        axis_perm: channel permutation to apply to pred (e.g. (2, 0, 1))
        axis_signs: (sx, sy, sz) sign flips to apply to pred before eval

    Returns:
        mean, median angular error (degrees) and accuracy at thresholds.
    """
    pred, pred_valid = _to_hwc_unit(pred)
    gt, gt_valid = _to_hwc_unit(gt)

    if valid_mask is None:
        valid_mask = gt_valid & pred_valid
    else:
        valid_mask = valid_mask & gt_valid & pred_valid

    if axis_perm is not None:
        pred = pred[:, :, list(axis_perm)]
    if axis_signs is not None:
        pred = pred * np.array(axis_signs, dtype=np.float32)

    valid_angles = _angular_error(pred, gt, valid_mask)
    if valid_angles.size < 10:
        return {}

    return {
        "mean": float(np.mean(valid_angles)),
        "median": float(np.median(valid_angles)),
        "rmse": float(np.sqrt(np.mean(valid_angles ** 2))),
        "acc_11.25": float((valid_angles < 11.25).mean()),
        "acc_22.5": float((valid_angles < 22.5).mean()),
        "acc_30": float((valid_angles < 30.0).mean()),
    }


# ======================================================================
# Dataset loader
# ======================================================================

def _discover_diode_samples(data_dir: Path) -> list[dict]:
    """Scan DIODE nested structure: scene_*/scan_*/*.png + *_depth.npy + *_depth_mask.npy."""
    samples = []
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for scan_dir in sorted(scene_dir.iterdir()):
            if not scan_dir.is_dir() or not scan_dir.name.startswith("scan_"):
                continue
            for img_path in sorted(scan_dir.glob("*.png")):
                stem = img_path.stem
                depth_path = scan_dir / f"{stem}_depth.npy"
                mask_path = scan_dir / f"{stem}_depth_mask.npy"
                if depth_path.exists():
                    samples.append({
                        "image": img_path,
                        "depth": depth_path,
                        "depth_mask": mask_path if mask_path.exists() else None,
                        "normals": None,
                        "normals_mask": None,
                    })
    return samples


def _discover_nyuv2_samples(data_dir: Path) -> list[dict]:
    """Scan NYUv2 flat structure: images/*.png + depth/*.npy + normals/*.npy + normals_mask/*.npy."""
    img_dir = data_dir / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"NYUv2 image directory not found: expected {data_dir}/images/"
        )

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        depth_path = data_dir / "depth" / f"{stem}.npy"
        normal_path = data_dir / "normals" / f"{stem}.npy"
        normal_mask_path = data_dir / "normals_mask" / f"{stem}.npy"
        samples.append({
            "image": img_path,
            "depth": depth_path if depth_path.exists() else None,
            "depth_mask": None,
            "normals": normal_path if normal_path.exists() else None,
            "normals_mask": normal_mask_path if normal_mask_path.exists() else None,
        })
    return samples


def discover_samples(data_dir: Path, dataset: str = "auto") -> list[dict]:
    """Discover dataset samples. Auto-detects format from directory structure."""
    if dataset == "auto":
        has_scene = any(d.name.startswith("scene_") for d in data_dir.iterdir() if d.is_dir())
        dataset = "diode" if has_scene else "nyuv2"
        logger.info("Auto-detected dataset format: %s", dataset)

    if dataset == "diode":
        return _discover_diode_samples(data_dir)
    else:
        return _discover_nyuv2_samples(data_dir)


def load_sample(sample: dict):
    """Load a single sample: image, depth, depth_mask, normals, normals_mask."""
    img = Image.open(sample["image"]).convert("RGB")

    depth = np.load(sample["depth"]) if sample["depth"] else None
    depth_mask = np.load(sample["depth_mask"]) if sample.get("depth_mask") else None
    normals = np.load(sample["normals"]) if sample.get("normals") else None
    normals_mask = np.load(sample["normals_mask"]) if sample.get("normals_mask") else None

    if depth is not None and depth.ndim == 3:
        depth = depth.squeeze()
    if depth_mask is not None and depth_mask.ndim == 3:
        depth_mask = depth_mask.squeeze()

    if normals is not None and normals_mask is None:
        norm_len = np.linalg.norm(normals, axis=0)
        normals_mask = norm_len > 0.1

    return img, depth, depth_mask, normals, normals_mask


# ======================================================================
# Evaluation loops
# ======================================================================

def _shard_indices(total: int, shard_id: int, n_shards: int,
                   max_samples: int | None = None) -> list[int]:
    """Return the sample indices for this shard."""
    indices = list(range(total))
    if max_samples:
        indices = indices[:max_samples]
    return indices[shard_id::n_shards]


def eval_depth(learner, samples: list[dict], indices: list[int],
               output_dir: Path | None,
               min_depth: float = 1e-3, max_depth: float = 10.0,
               prompt_file: str | None = None):
    logger.info("=== Depth evaluation: %d samples ===", len(indices))

    # Custom prompt + matching decode strategy
    custom_prompt = None
    if prompt_file:
        custom_prompt = Path(prompt_file).read_text().strip()
        from vision_learner.depth import DepthCodec
        custom_codec = DepthCodec()
        logger.info("Custom depth prompt: %s", prompt_file)

    all_raw = []
    all_si = []
    all_affine = []

    if output_dir:
        vis_dir = output_dir / "depth_vis"
        pred_dir = output_dir / "depth_pred"
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        if custom_prompt:
            (output_dir / "prompt_depth.txt").write_text(custom_prompt, encoding="utf-8")
        else:
            from vision_learner.prompt_loader import depth_prompt
            (output_dir / "prompt_depth.txt").write_text(depth_prompt(), encoding="utf-8")

    for i in tqdm(indices, desc="Depth"):
        img, gt_depth, gt_depth_mask, _, _ = load_sample(samples[i])
        if gt_depth is None:
            continue

        pred_path = pred_dir / f"{i:04d}.npy" if output_dir else None

        if pred_path and pred_path.exists():
            pred_depth = np.load(pred_path)
        elif custom_prompt:
            learner.load_model()
            vis = learner._generate(img, custom_prompt)
            vis_resized = vis.resize(img.size, Image.BILINEAR)
            pred_depth = custom_codec.decode_from_image(vis_resized)
            if output_dir:
                np.save(pred_path, pred_depth)
                vis.save(vis_dir / f"{i:04d}_vis.png")
        else:
            pred_depth, vis = learner.estimate_depth(img)
            if output_dir:
                np.save(pred_path, pred_depth)
                vis.save(vis_dir / f"{i:04d}_vis.png")

        if gt_depth_mask is not None:
            invalid = gt_depth_mask == 0
            gt_depth = gt_depth.copy()
            gt_depth[invalid] = 0

        if pred_depth.shape != gt_depth.shape:
            from scipy.ndimage import zoom as _zoom
            pred_resized = _zoom(
                pred_depth.astype(np.float32),
                (gt_depth.shape[0] / pred_depth.shape[0],
                 gt_depth.shape[1] / pred_depth.shape[1]),
                order=1,
            )
        else:
            pred_resized = pred_depth

        if output_dir:
            import matplotlib.cm as cm
            h, w = gt_depth.shape
            img_np = np.array(img.resize((w, h), Image.BILINEAR))
            def _depth_to_rgb(d, lo=None, hi=None):
                valid = np.isfinite(d) & (d > 0)
                if not valid.any():
                    return np.full(d.shape + (3,), 128, dtype=np.uint8)
                if lo is None:
                    lo = np.percentile(d[valid], 2)
                if hi is None:
                    hi = np.percentile(d[valid], 98)
                d_clip = np.clip(d, lo, hi)
                d_norm = (d_clip - lo) / max(hi - lo, 1e-8)
                rgb = (cm.plasma(d_norm)[:, :, :3] * 255).astype(np.uint8)
                rgb[~valid] = 128
                return rgb
            gt_valid = gt_depth > 0
            gt_lo = np.percentile(gt_depth[gt_valid], 2) if gt_valid.any() else 0
            gt_hi = np.percentile(gt_depth[gt_valid], 98) if gt_valid.any() else 1
            gt_rgb = _depth_to_rgb(gt_depth, gt_lo, gt_hi)
            if gt_valid.sum() >= 10:
                p = pred_resized[gt_valid].flatten()
                g = gt_depth[gt_valid].flatten()
                A = np.vstack([p, np.ones_like(p)]).T
                res = np.linalg.lstsq(A, g, rcond=None)
                va, vb = res[0]
                pred_vis = va * pred_resized + vb
            else:
                pred_vis = pred_resized
            pred_rgb = _depth_to_rgb(pred_vis, gt_lo, gt_hi)
            compare = np.concatenate([img_np, gt_rgb, pred_rgb], axis=1)
            Image.fromarray(compare).save(vis_dir / f"{i:04d}_compare.png")

        m_raw = depth_metrics(pred_resized, gt_depth, min_depth=min_depth, max_depth=max_depth)
        m_si = depth_metrics_scale_invariant(pred_resized, gt_depth, min_depth=min_depth, max_depth=max_depth)
        m_aff = depth_metrics_affine_invariant(pred_resized, gt_depth, min_depth=min_depth, max_depth=max_depth)

        if m_raw:
            all_raw.append(m_raw)
        if m_si:
            all_si.append(m_si)
        if m_aff:
            all_affine.append(m_aff)

    if not all_raw:
        logger.warning("No valid depth samples!")
        return {}

    def _avg(lst):
        return {k: float(np.mean([m[k] for m in lst])) for k in lst[0]}

    avg_raw = _avg(all_raw)
    avg_si = _avg(all_si) if all_si else {}
    avg_aff = _avg(all_affine) if all_affine else {}

    logger.info("--- Depth (raw, no alignment) ---")
    logger.info("  δ₁↑=%.4f  δ₂↑=%.4f  δ₃↑=%.4f", avg_raw["delta1"], avg_raw["delta2"], avg_raw["delta3"])
    logger.info("  AbsRel↓=%.4f  RMSE↓=%.4f", avg_raw["abs_rel"], avg_raw["rmse"])

    if avg_si:
        logger.info("--- Depth (median-scaled) ---")
        logger.info("  δ₁↑=%.4f  δ₂↑=%.4f  δ₃↑=%.4f", avg_si["delta1"], avg_si["delta2"], avg_si["delta3"])
        logger.info("  AbsRel↓=%.4f  RMSE↓=%.4f  scale=%.4f", avg_si["abs_rel"], avg_si["rmse"], avg_si["scale_factor"])

    if avg_aff:
        logger.info("--- Depth (affine-aligned, best for zero-shot) ---")
        logger.info("  δ₁↑=%.4f  δ₂↑=%.4f  δ₃↑=%.4f", avg_aff["delta1"], avg_aff["delta2"], avg_aff["delta3"])
        logger.info("  AbsRel↓=%.4f  RMSE↓=%.4f  a=%.4f b=%.4f", avg_aff["abs_rel"], avg_aff["rmse"], avg_aff["affine_a"], avg_aff["affine_b"])

    return {
        "depth_raw": avg_raw,
        "depth_scale_inv": avg_si,
        "depth_affine": avg_aff,
        "_per_sample_depth": all_raw,
        "_per_sample_depth_si": all_si,
        "_per_sample_depth_affine": all_affine,
    }


def eval_normals(learner, samples: list[dict], indices: list[int],
                 output_dir: Path | None):
    logger.info("=== Normals evaluation: %d samples ===", len(indices))

    if output_dir:
        vis_dir = output_dir / "normals_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        from vision_learner.prompt_loader import normals_prompt
        (output_dir / "prompt_normals.txt").write_text(normals_prompt(), encoding="utf-8")

    # --- Auto-detect axis convention on first N_CALIB samples ---
    # Each (model, dataset) pair may use different normal coordinate systems.
    # We try all 48 combinations (6 permutations × 8 sign flips) on a few
    # samples and pick the one with lowest mean angular error.
    N_CALIB = min(5, len(indices))
    convention_cache = output_dir / "normal_convention.json" if output_dir else None

    if convention_cache and convention_cache.exists():
        with open(convention_cache) as f:
            conv = json.load(f)
        axis_perm = tuple(conv["perm"])
        axis_signs = tuple(conv["signs"])
        logger.info("Loaded cached axis convention: perm=%s signs=%s (mean=%.2f°)",
                     axis_perm, axis_signs, conv["calib_mean"])
    else:
        logger.info("Calibrating axis convention on %d samples ...", N_CALIB)
        import itertools
        all_perms = list(itertools.permutations(range(3)))
        combo_errors = {}

        calib_pred_dir = output_dir / "normals_pred" if output_dir else None
        if calib_pred_dir:
            calib_pred_dir.mkdir(parents=True, exist_ok=True)

        for calib_idx in indices[:N_CALIB]:
            img, _, _, gt_normals, gt_normals_mask = load_sample(samples[calib_idx])
            if gt_normals is None:
                continue

            calib_pred_path = calib_pred_dir / f"{calib_idx:04d}.npy" if calib_pred_dir else None
            if calib_pred_path and calib_pred_path.exists():
                pred_normals = np.load(calib_pred_path)
            else:
                pred_normals, vis = learner.estimate_normals(img)
                if calib_pred_dir:
                    np.save(calib_pred_path, pred_normals)
                    vis_dir = output_dir / "normals_vis"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis.save(vis_dir / f"{calib_idx:04d}_vis.png")

            pred_hwc, pred_valid = _to_hwc_unit(pred_normals)
            gt_hwc, gt_valid = _to_hwc_unit(gt_normals)

            if pred_hwc.shape[:2] != gt_hwc.shape[:2]:
                from scipy.ndimage import zoom as _zoom
                h, w = gt_hwc.shape[:2]
                pred_hwc = _zoom(pred_hwc, (h / pred_hwc.shape[0], w / pred_hwc.shape[1], 1), order=1)
                pred_valid = np.ones(gt_hwc.shape[:2], dtype=bool)

            calib_valid = gt_valid & pred_valid
            if gt_normals_mask is not None:
                calib_valid &= gt_normals_mask
            for perm in all_perms:
                reordered = pred_hwc[:, :, list(perm)]
                for sx, sy, sz in itertools.product([1, -1], repeat=3):
                    key = (perm, (sx, sy, sz))
                    flipped = reordered * np.array([sx, sy, sz], dtype=np.float32)
                    angles = _angular_error(flipped, gt_hwc, calib_valid)
                    if angles.size == 0:
                        continue
                    combo_errors.setdefault(key, []).append(float(np.mean(angles)))

        if not combo_errors:
            logger.warning("No valid calibration samples; falling back to identity axes")
            axis_perm = (0, 1, 2)
            axis_signs = (1, 1, 1)
            calib_mean = float("nan")
        else:
            best_key = min(combo_errors, key=lambda k: np.mean(combo_errors[k]))
            axis_perm, axis_signs = best_key
            calib_mean = float(np.mean(combo_errors[best_key]))
        logger.info("Best axis convention: perm=%s signs=%s (calib mean=%.2f°)",
                     axis_perm, axis_signs, calib_mean)

        if convention_cache:
            with open(convention_cache, "w") as f:
                json.dump({"perm": list(axis_perm), "signs": list(axis_signs),
                           "calib_mean": calib_mean}, f)

    # --- Evaluate all samples with detected convention ---
    all_metrics = []
    pred_dir = output_dir / "normals_pred" if output_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(indices, desc="Normals"):
        img, _, _, gt_normals, gt_normals_mask = load_sample(samples[i])
        if gt_normals is None:
            continue

        pred_path = pred_dir / f"{i:04d}.npy" if pred_dir else None

        if pred_path and pred_path.exists():
            pred_normals = np.load(pred_path)
        else:
            pred_normals, vis = learner.estimate_normals(img)
            if output_dir:
                np.save(pred_path, pred_normals)
                vis.save(vis_dir / f"{i:04d}_vis.png")

        pred_hwc, pred_valid = _to_hwc_unit(pred_normals)
        gt_hwc, gt_valid = _to_hwc_unit(gt_normals)

        if pred_hwc.shape[:2] != gt_hwc.shape[:2]:
            from scipy.ndimage import zoom as _zoom
            h, w = gt_hwc.shape[:2]
            pred_hwc = _zoom(pred_hwc, (h / pred_hwc.shape[0], w / pred_hwc.shape[1], 1), order=1)
            pred_valid = np.ones(gt_hwc.shape[:2], dtype=bool)

        eval_mask = gt_valid & pred_valid
        if gt_normals_mask is not None:
            eval_mask &= gt_normals_mask

        if output_dir:
            h, w = gt_hwc.shape[:2]
            img_np = np.array(img.resize((w, h), Image.BILINEAR))
            def _normals_to_rgb(n):
                return ((n + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            perm = list(axis_perm)
            signs = list(axis_signs)
            pred_calibrated = pred_hwc[:, :, perm] * np.array(signs, dtype=np.float32)
            compare = np.concatenate([img_np, _normals_to_rgb(gt_hwc), _normals_to_rgb(pred_calibrated)], axis=1)
            Image.fromarray(compare).save(vis_dir / f"{i:04d}_compare.png")

        m = normal_metrics(pred_hwc, gt_normals,
                           valid_mask=eval_mask,
                           axis_perm=axis_perm, axis_signs=axis_signs)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        logger.warning("No valid normal samples!")
        return {}

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}

    logger.info("--- Surface Normals (perm=%s, signs=%s) ---", axis_perm, axis_signs)
    logger.info("  Mean↓=%.2f°  Median↓=%.2f°  RMSE↓=%.2f°", avg["mean"], avg["median"], avg["rmse"])
    logger.info("  <11.25°↑=%.4f  <22.5°↑=%.4f  <30°↑=%.4f", avg["acc_11.25"], avg["acc_22.5"], avg["acc_30"])

    result = {"normals": avg, "_per_sample_normals": all_metrics}
    result["normals"]["axis_perm"] = list(axis_perm)
    result["normals"]["axis_signs"] = list(axis_signs)
    return result


# ======================================================================
# Main
# ======================================================================

def merge_shard_results(output_dir: Path, n_shards: int,
                        task: str = "all") -> None:
    """Merge per-shard JSON results into a single results file."""
    all_depth = []
    all_depth_si = []
    all_depth_aff = []
    all_normals = []
    meta = None

    task_suffix = f"_{task}" if task != "all" else ""

    for sid in range(n_shards):
        p = output_dir / f"results_shard{sid}{task_suffix}.json"
        if not p.exists():
            p_fallback = output_dir / f"results_shard{sid}.json"
            if p_fallback.exists():
                p = p_fallback
            else:
                logger.warning("Missing shard %d: %s", sid, p)
                continue
        with open(p) as f:
            data = json.load(f)
        if meta is None:
            meta = data.get("meta", {})
        all_depth.extend(data.get("_per_sample_depth", []))
        all_depth_si.extend(data.get("_per_sample_depth_si", []))
        all_depth_aff.extend(data.get("_per_sample_depth_affine", []))
        all_normals.extend(data.get("_per_sample_normals", []))

    def _avg(lst):
        return {k: float(np.mean([m[k] for m in lst])) for k in lst[0]}

    results = {}
    if all_depth:
        results["depth_raw"] = _avg(all_depth)
    if all_depth_si:
        results["depth_scale_inv"] = _avg(all_depth_si)
    if all_depth_aff:
        results["depth_affine"] = _avg(all_depth_aff)
    if all_normals:
        results["normals"] = _avg(all_normals)
        conv_path = output_dir / "normal_convention.json"
        if conv_path.exists():
            with open(conv_path) as f:
                conv = json.load(f)
            results["normals"]["axis_perm"] = conv.get("perm")
            results["normals"]["axis_signs"] = conv.get("signs")

    if meta:
        meta["n_depth_samples"] = len(all_depth)
        meta["n_normal_samples"] = len(all_normals)
        meta["n_samples"] = max(len(all_depth), len(all_normals))
        meta["n_shards"] = n_shards
        results["meta"] = meta

    out_path = output_dir / f"results{task_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=== Merged results from %d shards (task=%s) ===", n_shards, task)
    if "depth_raw" in results:
        d = results["depth_raw"]
        logger.info("Depth (raw):       δ₁=%.4f  AbsRel=%.4f  RMSE=%.4f", d["delta1"], d["abs_rel"], d["rmse"])
    if "depth_scale_inv" in results:
        d = results["depth_scale_inv"]
        logger.info("Depth (scaled):    δ₁=%.4f  AbsRel=%.4f  RMSE=%.4f", d["delta1"], d["abs_rel"], d["rmse"])
    if "depth_affine" in results:
        d = results["depth_affine"]
        logger.info("Depth (affine):    δ₁=%.4f  AbsRel=%.4f  RMSE=%.4f", d["delta1"], d["abs_rel"], d["rmse"])
    if "normals" in results:
        n = results["normals"]
        logger.info("Normals:           Mean=%.2f°  Median=%.2f°  <11.25°=%.4f  <22.5°=%.4f  <30°=%.4f",
                     n["mean"], n["median"], n["acc_11.25"], n["acc_22.5"], n["acc_30"])
    logger.info("Saved → %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on NYUv2 val set")
    parser.add_argument("--model", required=True, help="Model path or HF id")
    parser.add_argument("--backend", default="qwen",
                        choices=["qwen", "firered", "longcat", "omnigen2",
                                 "kontext", "hunyuan", "generic"])
    parser.add_argument("--data", required=True, help="Path to data/nyuv2/")
    parser.add_argument("--task", default="all", choices=["all", "depth", "normals"])
    parser.add_argument("--output", "-o", default="output/eval_nyuv2",
                        help="Output directory for results and visualisations")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: per-backend, see DEFAULT_INFERENCE_PARAMS)")
    parser.add_argument("--cfg", type=float, default=None,
                        help="True CFG scale (default: per-backend, see DEFAULT_INFERENCE_PARAMS)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Max valid depth in metres (NYUv2=10, DIODE indoor=10, DIODE outdoor=350)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard index (0-based) for multi-GPU parallel eval")
    parser.add_argument("--n_shards", type=int, default=1,
                        help="Total number of shards (= number of GPUs)")
    parser.add_argument("--prompt", default=None,
                        help="Custom depth prompt file (e.g. prompts/depth_grayscale.txt)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard results instead of running eval")
    args = parser.parse_args()

    data_dir = Path(args.data)

    model_tag = Path(args.model).name

    output_dir = Path(args.output) / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shard_results(output_dir, args.n_shards, task=args.task)
        return

    samples = discover_samples(data_dir)
    n_total = len(samples)

    shard_id = args.shard_id if args.shard_id is not None else 0
    n_shards = args.n_shards

    device = args.device

    indices = _shard_indices(n_total, shard_id, n_shards, args.max_samples)
    logger.info("Data: %s (%d total, shard %d/%d → %d samples, device=%s)",
                data_dir, n_total, shard_id, n_shards, len(indices), device)

    from vision_learner.pipeline import VisionLearner, DEFAULT_INFERENCE_PARAMS
    defaults = DEFAULT_INFERENCE_PARAMS.get(args.backend, {})
    steps = args.steps if args.steps is not None else defaults.get("num_inference_steps", 40)
    cfg = args.cfg if args.cfg is not None else defaults.get("true_cfg_scale", 4.0)
    logger.info("Inference params: steps=%d, cfg=%.1f (backend=%s)", steps, cfg, args.backend)
    learner = VisionLearner(
        model_name=args.model,
        backend=args.backend,
        device=device,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        seed=args.seed,
    )

    results = {}
    t0 = time.time()

    if args.task in ("all", "depth"):
        results.update(eval_depth(
            learner, samples, indices, output_dir,
            min_depth=args.min_depth, max_depth=args.max_depth,
            prompt_file=args.prompt,
        ))

    if args.task in ("all", "normals"):
        results.update(eval_normals(
            learner, samples, indices, output_dir,
        ))

    elapsed = time.time() - t0
    results["meta"] = {
        "model": args.model,
        "backend": args.backend,
        "n_samples": len(indices),
        "shard_id": shard_id,
        "n_shards": n_shards,
        "task": args.task,
        "prompt": args.prompt,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "steps": steps,
        "cfg": cfg,
        "seed": args.seed,
        "elapsed_sec": round(elapsed, 1),
    }

    task_suffix = f"_{args.task}" if args.task != "all" else ""
    if n_shards > 1:
        results_path = output_dir / f"results_shard{shard_id}{task_suffix}.json"
    else:
        results_path = output_dir / f"results{task_suffix}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", results_path)
    logger.info("Shard %d done: %.1f sec (%.1f sec/sample)",
                shard_id, elapsed, elapsed / max(len(indices), 1))


if __name__ == "__main__":
    main()
