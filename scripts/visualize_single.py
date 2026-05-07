"""
Generate paper-quality visualization for a single image + task.

Outputs:
  <out_dir>/input.png          — original input image
  <out_dir>/model_output.png   — raw model generation (vis)
  <out_dir>/decoded.png        — decoded result (depth map / normal map / seg mask)
  <out_dir>/ground_truth.png   — GT visualized in the same style
  <out_dir>/compare.png        — all four side-by-side

Usage:
  # Depth on NYUv2 sample index 19
  python scripts/visualize_single.py --task depth \
      --model /path/to/Qwen-Image-Edit-2511 --backend qwen \
      --dataset nyuv2 --data data/nyuv2 --index 19

  # Normals on NYUv2 sample index 5
  python scripts/visualize_single.py --task normals \
      --model /path/to/model --backend qwen \
      --dataset nyuv2 --data data/nyuv2 --index 5

  # Segmentation (19-class) on Cityscapes sample index 0
  python scripts/visualize_single.py --task seg19 \
      --model /path/to/model --backend qwen \
      --dataset cityscapes --data data/Cityscapes --index 0

  # Segmentation (7-cat) on Cityscapes sample index 0
  python scripts/visualize_single.py --task seg7 \
      --model /path/to/model --backend qwen \
      --dataset cityscapes --data data/Cityscapes --index 0
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

import numpy as np
import matplotlib.cm as cm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Data loaders ──────────────────────────────────────────────────────

def _load_nyuv2(data_dir: Path, index: int):
    img_dir = data_dir / "images"
    normal_dir = data_dir / "normals"
    imgs = sorted(img_dir.glob("*.png"))

    if index >= len(imgs):
        raise ValueError(f"Index {index} out of range (total {len(imgs)})")
    img_path = imgs[index]
    stem = img_path.stem
    depth_path = data_dir / "depth" / f"{stem}.npy"
    normal_path = normal_dir / f"{stem}.npy"

    img = Image.open(img_path).convert("RGB")
    gt_depth = np.load(depth_path) if depth_path.exists() else None
    gt_normals = np.load(normal_path) if normal_path.exists() else None
    if gt_depth is not None and gt_depth.ndim == 3:
        gt_depth = gt_depth.squeeze()
    return img, gt_depth, gt_normals


def _load_diode(data_dir: Path, index: int):
    samples = []
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for scan_dir in sorted(scene_dir.iterdir()):
            if not scan_dir.is_dir() or not scan_dir.name.startswith("scan_"):
                continue
            for img_path in sorted(scan_dir.glob("*.png")):
                depth_path = scan_dir / f"{img_path.stem}_depth.npy"
                if depth_path.exists():
                    samples.append((img_path, depth_path))
    if index >= len(samples):
        raise ValueError(f"Index {index} out of range (total {len(samples)})")
    img_path, depth_path = samples[index]
    img = Image.open(img_path).convert("RGB")
    gt_depth = np.load(depth_path).squeeze()
    return img, gt_depth, None


def _load_cityscapes(data_dir: Path, index: int):
    img_root = data_dir / "leftImg8bit" / "val"
    gt_root = data_dir / "gtFine" / "val"
    pairs = []
    for city in sorted(img_root.iterdir()):
        if not city.is_dir():
            continue
        for ip in sorted(city.glob("*_leftImg8bit.png")):
            stem = ip.stem.replace("_leftImg8bit", "")
            gp = gt_root / city.name / f"{stem}_gtFine_labelIds.png"
            if gp.exists():
                pairs.append((ip, gp))
    if index >= len(pairs):
        raise ValueError(f"Index {index} out of range (total {len(pairs)})")
    img_path, gt_path = pairs[index]
    img = Image.open(img_path).convert("RGB")
    gt_label = np.array(Image.open(gt_path), dtype=np.int32)
    return img, gt_label


# ── Cityscapes constants ──────────────────────────────────────────────

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
IGNORE = 255

PALETTE = [
    ("red",          (255,   0,   0)),
    ("yellow",       (255, 255,   0)),
    ("orange",       (255, 165,   0)),
    ("purple",       (128,   0, 128)),
    ("brown",        (139,  69,  19)),
    ("magenta",      (255,   0, 255)),
    ("dark yellow",  (200, 200,   0)),
    ("pink",         (255, 192, 203)),
    ("green",        (  0, 255,   0)),
    ("light green",  (144, 238, 144)),
    ("cyan",         (  0, 255, 255)),
    ("blue",         (  0,   0, 255)),
    ("dark blue",    (  0,   0, 128)),
    ("white",        (255, 255, 255)),
    ("gray",         (128, 128, 128)),
    ("dark green",   (  0, 128,   0)),
    ("dark red",     (128,   0,   0)),
    ("light blue",   (128, 128, 255)),
    ("light pink",   (255, 182, 193)),
]

VIS_COLOURS = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
    (0,60,100),(0,80,100),(0,0,230),(119,11,32),
]

TRAINID_TO_7CAT = np.array([0,0,1,1,1,2,2,2,3,3,4,5,5,6,6,6,6,6,6], dtype=np.int32)
CAT7_NAMES = ["flat","construction","object","nature","sky","human","vehicle"]
CAT7_PALETTE = [
    ("red",     (255,   0,   0)),
    ("orange",  (255, 165,   0)),
    ("magenta", (255,   0, 255)),
    ("green",   (  0, 255,   0)),
    ("cyan",    (  0, 255, 255)),
    ("blue",    (  0,   0, 255)),
    ("yellow",  (255, 255,   0)),
]
CAT7_VIS = [(128,64,128),(70,70,70),(153,153,153),(107,142,35),(70,130,180),(220,20,60),(0,0,142)]

_LID2TID = np.array([
    255,255,255,255,255,255,255,
    0,1,255,255,
    2,3,4,255,255,255,
    5,255,6,7,
    8,9,10,
    11,12,13,14,15,255,255,
    16,17,18,
], dtype=np.uint8)

_CAT7_GROUP_NAMES = {
    0: "roads and sidewalks",
    1: "buildings, walls, and fences",
    2: "poles and traffic signs",
    3: "trees and grass",
    4: "sky",
    5: "people",
    6: "vehicles",
}


def _label_to_train(label_ids):
    out = np.full_like(label_ids, IGNORE, dtype=np.uint8)
    valid = label_ids < len(_LID2TID)
    out[valid] = _LID2TID[label_ids[valid]]
    return out.astype(np.int32)


# ── Visualization helpers ─────────────────────────────────────────────

def depth_to_rgb(d, lo=None, hi=None):
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


def normals_to_rgb(n):
    if n.shape[0] == 3 and n.ndim == 3:
        n = np.transpose(n, (1, 2, 0))
    return ((n + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


def seg_to_rgb(mask, colours):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(colours):
        rgb[mask == i] = c
    return rgb


# ── Task runners ──────────────────────────────────────────────────────

def run_depth(learner, img, gt_depth, out_dir, prompt_file=None):
    from vision_learner.depth import DepthCodec

    if prompt_file:
        prompt = Path(prompt_file).read_text().strip()
        logger.info("Depth prompt: %s", prompt)
        codec = DepthCodec()
        learner.load_model()
        vis = learner._generate(img, prompt)
        vis_resized = vis.resize(img.size, Image.BILINEAR)
        pred_depth = codec.decode_from_image(vis_resized)
    else:
        pred_depth, vis = learner.estimate_depth(img)
        vis_resized = vis.resize(img.size, Image.BILINEAR)

    if pred_depth.shape != gt_depth.shape:
        from scipy.ndimage import zoom as _zoom
        pred_depth = _zoom(
            pred_depth.astype(np.float32),
            (gt_depth.shape[0] / pred_depth.shape[0],
             gt_depth.shape[1] / pred_depth.shape[1]),
            order=1,
        )

    # Affine alignment: pred_aligned = a * pred + b
    gt_valid = gt_depth > 0
    if gt_valid.any():
        p = pred_depth[gt_valid].flatten()
        g = gt_depth[gt_valid].flatten()
        A = np.vstack([p, np.ones_like(p)]).T
        result = np.linalg.lstsq(A, g, rcond=None)
        a, b = result[0]
        pred_aligned = a * pred_depth + b
    else:
        pred_aligned = pred_depth

    gt_lo = np.percentile(gt_depth[gt_valid], 2) if gt_valid.any() else 0
    gt_hi = np.percentile(gt_depth[gt_valid], 98) if gt_valid.any() else 1

    h, w = gt_depth.shape
    img_resized = np.array(img.resize((w, h), Image.BILINEAR))

    vis.save(out_dir / "model_output.png")
    Image.fromarray(depth_to_rgb(pred_aligned, gt_lo, gt_hi)).save(out_dir / "decoded.png")
    Image.fromarray(depth_to_rgb(gt_depth, gt_lo, gt_hi)).save(out_dir / "ground_truth.png")

    compare = np.concatenate([
        img_resized,
        depth_to_rgb(gt_depth, gt_lo, gt_hi),
        np.array(vis_resized.resize((w, h), Image.BILINEAR)),
        depth_to_rgb(pred_aligned, gt_lo, gt_hi),
    ], axis=1)
    Image.fromarray(compare).save(out_dir / "compare.png")
    logger.info("Depth visualization saved to %s", out_dir)


def run_normals(learner, img, gt_normals, out_dir):
    pred_normals, vis = learner.estimate_normals(img)
    vis_resized = vis.resize(img.size, Image.BILINEAR)

    pred_hwc = pred_normals
    if pred_hwc.shape[0] == 3 and pred_hwc.ndim == 3:
        pred_hwc = np.transpose(pred_hwc, (1, 2, 0))
    gt_hwc = gt_normals
    if gt_hwc.shape[0] == 3 and gt_hwc.ndim == 3:
        gt_hwc = np.transpose(gt_hwc, (1, 2, 0))

    if pred_hwc.shape[:2] != gt_hwc.shape[:2]:
        from scipy.ndimage import zoom as _zoom
        h, w = gt_hwc.shape[:2]
        pred_hwc = _zoom(pred_hwc, (h / pred_hwc.shape[0], w / pred_hwc.shape[1], 1), order=1)

    h, w = gt_hwc.shape[:2]
    img_resized = np.array(img.resize((w, h), Image.BILINEAR))

    vis.save(out_dir / "model_output.png")
    Image.fromarray(normals_to_rgb(pred_hwc)).save(out_dir / "decoded.png")
    Image.fromarray(normals_to_rgb(gt_hwc)).save(out_dir / "ground_truth.png")

    compare = np.concatenate([
        img_resized,
        normals_to_rgb(gt_hwc),
        np.array(vis_resized.resize((w, h), Image.BILINEAR)),
        normals_to_rgb(pred_hwc),
    ], axis=1)
    Image.fromarray(compare).save(out_dir / "compare.png")
    logger.info("Normals visualization saved to %s", out_dir)


def run_seg19(learner, img, gt_label, out_dir):
    from vision_learner.prompt_loader import render_prompt
    gt = _label_to_train(gt_label)
    h, w = gt.shape
    present = sorted(set(gt.ravel()) - {IGNORE})
    parts = [f"the {CLASSES[t]} {PALETTE[t][0]}" for t in present]
    prompt = render_prompt(
        "semantic_segmentation_19.txt",
        class_color_list=", ".join(parts),
    )

    logger.info("Prompt: %s", prompt)
    learner.load_model()
    vis = learner._generate(img, prompt)

    rgb = np.array(vis.resize((w, h), Image.NEAREST), dtype=np.float32)
    colours = np.array([PALETTE[t][1] for t in present] + [(0,0,0)], dtype=np.float32)
    dists = np.linalg.norm(rgb[:,:,None,:] - colours[None,None,:,:], axis=-1)
    best = np.argmin(dists, axis=-1)
    pred = np.full((h, w), IGNORE, dtype=np.int32)
    for j, tid in enumerate(present):
        pred[best == j] = tid

    img_resized = np.array(img.resize((w, h), Image.BILINEAR))

    vis.save(out_dir / "model_output.png")
    Image.fromarray(seg_to_rgb(pred, VIS_COLOURS)).save(out_dir / "decoded.png")
    Image.fromarray(seg_to_rgb(gt, VIS_COLOURS)).save(out_dir / "ground_truth.png")

    compare = np.concatenate([
        img_resized,
        seg_to_rgb(gt, VIS_COLOURS),
        np.array(vis.resize((w, h), Image.NEAREST)),
        seg_to_rgb(pred, VIS_COLOURS),
    ], axis=1)
    Image.fromarray(compare).save(out_dir / "compare.png")
    logger.info("Seg-19 visualization saved to %s", out_dir)


def run_seg7(learner, img, gt_label, out_dir):
    from vision_learner.prompt_loader import render_prompt
    gt = _label_to_train(gt_label)
    h, w = gt.shape
    gt_safe = np.where((gt >= 0) & (gt < len(TRAINID_TO_7CAT)), gt, 0)
    gt7 = np.where((gt >= 0) & (gt < len(TRAINID_TO_7CAT)),
                   TRAINID_TO_7CAT[gt_safe], IGNORE)

    present_train = sorted(set(gt.ravel()) - {IGNORE})
    present_cats = sorted(set(TRAINID_TO_7CAT[t] for t in present_train))
    parts = [f"all {_CAT7_GROUP_NAMES[c]} solid {CAT7_PALETTE[c][0]}"
             for c in present_cats]
    prompt = render_prompt(
        "semantic_segmentation_7.txt",
        category_color_list=", ".join(parts),
    )

    logger.info("Prompt: %s", prompt)
    learner.load_model()
    vis = learner._generate(img, prompt)

    rgb = np.array(vis.resize((w, h), Image.NEAREST), dtype=np.float32)
    colours = np.array([CAT7_PALETTE[c][1] for c in present_cats] + [(0,0,0)],
                       dtype=np.float32)
    dists = np.linalg.norm(rgb[:,:,None,:] - colours[None,None,:,:], axis=-1)
    best = np.argmin(dists, axis=-1)
    pred7 = np.full((h, w), IGNORE, dtype=np.int32)
    for j, cid in enumerate(present_cats):
        pred7[best == j] = cid

    img_resized = np.array(img.resize((w, h), Image.BILINEAR))

    vis.save(out_dir / "model_output.png")
    Image.fromarray(seg_to_rgb(pred7, CAT7_VIS)).save(out_dir / "decoded.png")
    Image.fromarray(seg_to_rgb(gt7, CAT7_VIS)).save(out_dir / "ground_truth.png")

    compare = np.concatenate([
        img_resized,
        seg_to_rgb(gt7, CAT7_VIS),
        np.array(vis.resize((w, h), Image.NEAREST)),
        seg_to_rgb(pred7, CAT7_VIS),
    ], axis=1)
    Image.fromarray(compare).save(out_dir / "compare.png")
    logger.info("Seg-7 visualization saved to %s", out_dir)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate single-image visualization for paper figures")
    parser.add_argument("--task", required=True, choices=["depth", "normals", "seg19", "seg7"])
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--backend", default="qwen",
                        choices=["qwen", "firered", "longcat", "omnigen2",
                                 "kontext", "hunyuan", "generic"])
    parser.add_argument("--dataset", default="nyuv2", choices=["nyuv2", "diode", "cityscapes"])
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--index", type=int, default=0, help="Sample index in the dataset")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: output/vis_single/<task>_<index>_<model>)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: per-backend)")
    parser.add_argument("--cfg", type=float, default=None,
                        help="True CFG scale (default: per-backend)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=None, help="Path to a prompt .txt file (overrides default for depth/normals)")
    args = parser.parse_args()

    model_tag = Path(args.model).name
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path(f"output/vis_single/{args.task}_{args.index:04d}_{model_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from vision_learner.pipeline import VisionLearner, DEFAULT_INFERENCE_PARAMS
    defaults = DEFAULT_INFERENCE_PARAMS.get(args.backend, {})
    steps = args.steps if args.steps is not None else defaults.get("num_inference_steps", 40)
    cfg = args.cfg if args.cfg is not None else defaults.get("true_cfg_scale", 4.0)
    logger.info("Inference params: steps=%d, cfg=%.1f (backend=%s)", steps, cfg, args.backend)
    learner = VisionLearner(
        model_name=args.model, backend=args.backend,
        device=args.device, num_inference_steps=steps,
        true_cfg_scale=cfg, seed=args.seed,
    )

    data_dir = Path(args.data)

    if args.task == "depth":
        if args.dataset == "nyuv2":
            img, gt_depth, _ = _load_nyuv2(data_dir, args.index)
        else:
            img, gt_depth, _ = _load_diode(data_dir, args.index)
        if gt_depth is None:
            raise FileNotFoundError(f"GT depth not found for sample index {args.index}")
        img.save(out_dir / "input.png")
        run_depth(learner, img, gt_depth, out_dir, prompt_file=args.prompt)

    elif args.task == "normals":
        img, _, gt_normals = _load_nyuv2(data_dir, args.index)
        if gt_normals is None:
            raise FileNotFoundError(f"GT normals not found for sample index {args.index}")
        img.save(out_dir / "input.png")
        run_normals(learner, img, gt_normals, out_dir)

    elif args.task in ("seg19", "seg7"):
        img, gt_label = _load_cityscapes(data_dir, args.index)
        img.save(out_dir / "input.png")
        if args.task == "seg19":
            run_seg19(learner, img, gt_label, out_dir)
        else:
            run_seg7(learner, img, gt_label, out_dir)

    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
