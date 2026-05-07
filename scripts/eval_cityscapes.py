"""
Evaluate zero-shot semantic segmentation on Cityscapes val set.

Usage:
    python scripts/eval_cityscapes.py --model /path/to/model --data data/Cityscapes --max_samples 5
    python scripts/eval_cityscapes.py --model /path/to/model --data data/Cityscapes
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_cityscapes.py --model ... --shard_id 0 --n_shards 8
    python scripts/eval_cityscapes.py --model ... --n_shards 8 --merge
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, json, logging, time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
IGNORE = 255

# Colour name + RGB for each of the 19 classes (fixed per class).
PALETTE = [
    ("red",          (255,   0,   0)),  # 0  road
    ("yellow",       (255, 255,   0)),  # 1  sidewalk
    ("orange",       (255, 165,   0)),  # 2  building
    ("purple",       (128,   0, 128)),  # 3  wall
    ("brown",        (139,  69,  19)),  # 4  fence
    ("magenta",      (255,   0, 255)),  # 5  pole
    ("dark yellow",  (200, 200,   0)),  # 6  traffic light
    ("pink",         (255, 192, 203)),  # 7  traffic sign
    ("green",        (  0, 255,   0)),  # 8  vegetation
    ("light green",  (144, 238, 144)),  # 9  terrain
    ("cyan",         (  0, 255, 255)),  # 10 sky
    ("blue",         (  0,   0, 255)),  # 11 person
    ("dark blue",    (  0,   0, 128)),  # 12 rider
    ("white",        (255, 255, 255)),  # 13 car
    ("gray",         (128, 128, 128)),  # 14 truck
    ("dark green",   (  0, 128,   0)),  # 15 bus
    ("dark red",     (128,   0,   0)),  # 16 train
    ("light blue",   (128, 128, 255)),  # 17 motorcycle
    ("light pink",   (255, 182, 193)),  # 18 bicycle
]

# Cityscapes official colours for GT visualisation
VIS_COLOURS = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
    (0,60,100),(0,80,100),(0,0,230),(119,11,32),
]

CAT7_NAMES = ["flat","construction","object","nature","sky","human","vehicle"]
TRAINID_TO_7CAT = np.array([0,0,1,1,1,2,2,2,3,3,4,5,5,6,6,6,6,6,6], dtype=np.int32)
CAT7_PALETTE = [
    ("red",     (255,   0,   0)),  # flat
    ("orange",  (255, 165,   0)),  # construction
    ("magenta", (255,   0, 255)),  # object
    ("green",   (  0, 255,   0)),  # nature
    ("cyan",    (  0, 255, 255)),  # sky
    ("blue",    (  0,   0, 255)),  # human
    ("yellow",  (255, 255,   0)),  # vehicle
]
CAT7_VIS = [(128,64,128),(70,70,70),(153,153,153),(107,142,35),(70,130,180),(220,20,60),(0,0,142)]

# labelId → trainId
# fmt: off
_LID2TID = np.array([
    255,255,255,255,255,255,255,
    0,1,255,255,
    2,3,4,255,255,255,
    5,255,6,7,
    8,9,10,
    11,12,13,14,15,255,255,
    16,17,18,
], dtype=np.uint8)
# fmt: on

def _label_to_train(label_ids: np.ndarray) -> np.ndarray:
    out = np.full_like(label_ids, IGNORE, dtype=np.uint8)
    valid = label_ids < len(_LID2TID)
    out[valid] = _LID2TID[label_ids[valid]]
    return out.astype(np.int32)

# ── Prompt ────────────────────────────────────────────────────────────

from vision_learner.prompt_loader import render_prompt

_CAT7_GROUP_NAMES = {
    0: "roads and sidewalks",
    1: "buildings, walls, and fences",
    2: "poles and traffic signs",
    3: "trees and grass",
    4: "sky",
    5: "people",
    6: "vehicles",
}

def _build_prompt(gt_mask: np.ndarray) -> tuple[str, list[int]]:
    """Per-image prompt listing only classes present in GT."""
    present = sorted(set(gt_mask.ravel()) - {IGNORE})
    parts = [f"the {CLASSES[t]} {PALETTE[t][0]}" for t in present]
    prompt = render_prompt(
        "semantic_segmentation_19.txt",
        class_color_list=", ".join(parts),
    )
    return prompt, present

def _build_prompt_7cat(gt_mask: np.ndarray) -> tuple[str, list[int]]:
    """Per-image prompt: group sub-classes by super-category."""
    present_train = sorted(set(gt_mask.ravel()) - {IGNORE})
    present_cats = sorted(set(TRAINID_TO_7CAT[t] for t in present_train))
    parts = [f"all {_CAT7_GROUP_NAMES[c]} solid {CAT7_PALETTE[c][0]}"
             for c in present_cats]
    prompt = render_prompt(
        "semantic_segmentation_7.txt",
        category_color_list=", ".join(parts),
    )
    return prompt, present_cats

def _decode_7cat(vis: Image.Image, w: int, h: int,
                 present_cats: list[int]) -> np.ndarray:
    rgb = np.array(vis.convert("RGB").resize((w, h), Image.NEAREST), dtype=np.float32)
    colours = np.array([CAT7_PALETTE[c][1] for c in present_cats] + [(0,0,0)],
                       dtype=np.float32)
    dists = np.linalg.norm(rgb[:,:,None,:] - colours[None,None,:,:], axis=-1)
    best = np.argmin(dists, axis=-1)
    mask = np.full((h, w), IGNORE, dtype=np.int32)
    for j, cid in enumerate(present_cats):
        mask[best == j] = cid
    return mask

# ── Decode ────────────────────────────────────────────────────────────

def _decode(vis: Image.Image, w: int, h: int,
            present: list[int]) -> np.ndarray:
    """Nearest-colour decode among present classes + black background."""
    rgb = np.array(vis.convert("RGB").resize((w, h), Image.NEAREST), dtype=np.float32)
    colours = np.array([PALETTE[t][1] for t in present] + [(0,0,0)],
                       dtype=np.float32)
    dists = np.linalg.norm(rgb[:,:,None,:] - colours[None,None,:,:], axis=-1)
    best = np.argmin(dists, axis=-1)
    mask = np.full((h, w), IGNORE, dtype=np.int32)
    for j, tid in enumerate(present):
        mask[best == j] = tid
    return mask

# ── Metrics ───────────────────────────────────────────────────────────

def _metrics_19(pred, gt):
    """Per-image intersection/union counts for global (dataset-level) mIoU."""
    valid = gt != IGNORE
    pv, gv = pred[valid], gt[valid]
    if len(gv) == 0:
        return {}
    return {
        "intersection": [int(((pv == c) & (gv == c)).sum()) for c in range(len(CLASSES))],
        "union":        [int(((pv == c) | (gv == c)).sum()) for c in range(len(CLASSES))],
        "correct": int((pv == gv).sum()),
        "total":   int(len(gv)),
    }


# ── Visualisation ─────────────────────────────────────────────────────

def _overlay(image, pred, gt):
    h, w = gt.shape
    img = np.array(image.resize((w,h), Image.BILINEAR))
    def col(m):
        c = np.zeros((h,w,3), dtype=np.uint8)
        for i in range(len(CLASSES)):
            c[m==i] = VIS_COLOURS[i]
        return c
    return Image.fromarray(np.concatenate([img, col(gt), col(pred)], axis=1))

def _overlay_7cat(image, pred, gt):
    h, w = gt.shape
    img = np.array(image.resize((w,h), Image.BILINEAR))
    def col(m):
        c = np.zeros((h,w,3), dtype=np.uint8)
        for i in range(7):
            c[m==i] = CAT7_VIS[i]
        return c
    return Image.fromarray(np.concatenate([img, col(gt), col(pred)], axis=1))

# ── Data loader ───────────────────────────────────────────────────────

def _load_pairs(data_dir: Path):
    img_root = data_dir / "leftImg8bit" / "val"
    gt_root  = data_dir / "gtFine" / "val"
    if not img_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_root}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_root}")
    pairs = []
    for city in sorted(img_root.iterdir()):
        if not city.is_dir(): continue
        for ip in sorted(city.glob("*_leftImg8bit.png")):
            stem = ip.stem.replace("_leftImg8bit", "")
            gp = gt_root / city.name / f"{stem}_gtFine_labelIds.png"
            if gp.exists():
                pairs.append((ip, gp))
    return pairs

def _shard(total, sid, ns, mx=None):
    idx = list(range(total))
    if mx is not None:
        idx = idx[:mx]
    return idx[sid::ns]

# ── Global aggregation ────────────────────────────────────────────────

def _agg_global(metrics, n_classes, class_names, label):
    """Dataset-level mIoU: accumulate intersection/union across all images."""
    total_inter = [0] * n_classes
    total_union = [0] * n_classes
    total_correct = 0
    total_pixels = 0
    for m in metrics:
        for c in range(n_classes):
            total_inter[c] += m["intersection"][c]
            total_union[c] += m["union"][c]
        total_correct += m["correct"]
        total_pixels += m["total"]

    per_class_iou = {}
    for c in range(n_classes):
        if total_union[c] > 0:
            per_class_iou[class_names[c]] = total_inter[c] / total_union[c]

    miou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0
    pacc = total_correct / max(total_pixels, 1)
    logger.info("--- %s (global, %d samples) ---", label, len(metrics))
    logger.info("  mIoU=%.4f  PixelAcc=%.4f  (%d classes with support)",
                miou, pacc, len(per_class_iou))
    return miou, pacc


def _agg_per_image(metrics, n_classes, class_names, label):
    """Per-image mIoU then averaged (legacy method)."""
    mious, paccs = [], []
    for m in metrics:
        per_class_iou = []
        for c in range(n_classes):
            if m["union"][c] > 0:
                per_class_iou.append(m["intersection"][c] / m["union"][c])
        mious.append(float(np.mean(per_class_iou)) if per_class_iou else 0.0)
        paccs.append(m["correct"] / max(m["total"], 1))

    miou = float(np.mean(mious))
    pacc = float(np.mean(paccs))
    logger.info("--- %s (per-image avg, %d samples) ---", label, len(metrics))
    logger.info("  mIoU=%.4f  PixelAcc=%.4f", miou, pacc)
    return miou, pacc


# ── Eval loop ─────────────────────────────────────────────────────────

def evaluate(learner, pairs, indices, out_dir, mode="both", miou="global"):
    logger.info("=== Semantic segmentation (%s): %d samples ===", mode, len(indices))
    run19 = mode in ("19", "both")
    run7  = mode in ("7", "both")
    d19 = d7 = None
    if out_dir:
        if run19:
            for s in ["seg_vis","seg_pred"]: (out_dir/s).mkdir(parents=True, exist_ok=True)
            d19 = (out_dir/"seg_vis", out_dir/"seg_pred")
        if run7:
            for s in ["seg7_vis","seg7_pred"]: (out_dir/s).mkdir(parents=True, exist_ok=True)
            d7 = (out_dir/"seg7_vis", out_dir/"seg7_pred")

    m19_all, m7_all = [], []
    prompt_logged_19 = prompt_logged_7 = False

    if run19:
        for i in tqdm(indices, desc="Seg-19"):
            img_path, gt_path = pairs[i]
            image = Image.open(img_path).convert("RGB")
            gt = _label_to_train(np.array(Image.open(gt_path), dtype=np.int32))
            h, w = gt.shape
            prompt, present = _build_prompt(gt)
            if not prompt_logged_19 and out_dir:
                (out_dir / "prompt_19_example.txt").write_text(prompt, encoding="utf-8")
                prompt_logged_19 = True
            pp = d19[1]/f"{i:04d}.npy" if d19 else None
            vp = d19[0]/f"{i:04d}_vis.png" if d19 else None
            if pp and pp.exists():
                pred = np.load(pp)
            else:
                if vp and vp.exists():
                    vis = Image.open(vp)
                else:
                    learner.load_model()
                    vis = learner._generate(image, prompt)
                    if vp: vis.save(vp)
                pred = _decode(vis, w, h, present)
                if pp: np.save(pp, pred)
                if d19: _overlay(image, pred, gt).save(d19[0]/f"{i:04d}_compare.png")
            m = _metrics_19(pred, gt)
            if m: m19_all.append(m)

    if run7:
        for i in tqdm(indices, desc="Seg-7"):
            img_path, gt_path = pairs[i]
            image = Image.open(img_path).convert("RGB")
            gt = _label_to_train(np.array(Image.open(gt_path), dtype=np.int32))
            h, w = gt.shape
            gt_safe = np.where((gt >= 0) & (gt < len(TRAINID_TO_7CAT)), gt, 0)
            gt7 = np.where((gt >= 0) & (gt < len(TRAINID_TO_7CAT)),
                           TRAINID_TO_7CAT[gt_safe], IGNORE)
            prompt7, present_cats = _build_prompt_7cat(gt)
            if not prompt_logged_7 and out_dir:
                (out_dir / "prompt_7cat_example.txt").write_text(prompt7, encoding="utf-8")
                prompt_logged_7 = True
            pp7 = d7[1]/f"{i:04d}.npy" if d7 else None
            vp7 = d7[0]/f"{i:04d}_vis.png" if d7 else None
            if pp7 and pp7.exists():
                pred7 = np.load(pp7)
            else:
                if vp7 and vp7.exists():
                    vis7 = Image.open(vp7)
                else:
                    learner.load_model()
                    vis7 = learner._generate(image, prompt7)
                    if vp7: vis7.save(vp7)
                pred7 = _decode_7cat(vis7, w, h, present_cats)
                if pp7: np.save(pp7, pred7)
                if d7: _overlay_7cat(image, pred7, gt7).save(d7[0]/f"{i:04d}_compare.png")
            valid = gt7 != IGNORE
            pv, gv = pred7[valid], gt7[valid]
            if len(gv) == 0: continue
            m7_all.append({
                "intersection": [int(((pv == c) & (gv == c)).sum()) for c in range(7)],
                "union":        [int(((pv == c) | (gv == c)).sum()) for c in range(7)],
                "correct": int((pv == gv).sum()),
                "total":   int(len(gv)),
            })

    if not m19_all and not m7_all:
        logger.warning("No valid samples!"); return {}

    agg = _agg_global if miou == "global" else _agg_per_image
    res = {}
    if m19_all:
        miou19, pacc19 = agg(m19_all, len(CLASSES), CLASSES, "19-class")
        res["segmentation"] = {"miou": miou19, "pixel_accuracy": pacc19}
        res["_per_sample_segmentation"] = m19_all
    if m7_all:
        miou7, pacc7 = agg(m7_all, 7, CAT7_NAMES, "7-category")
        res["segmentation_7cat"] = {"miou": miou7, "pixel_accuracy": pacc7}
        res["_per_sample_segmentation_7cat"] = m7_all
    return res

# ── Merge shards ──────────────────────────────────────────────────────

def merge(out_dir, n_shards, mode="both", miou="global"):
    s19, s7, meta = [], [], None
    suffixes = []
    if mode in ("19","both"): suffixes.append("_19")
    if mode in ("7","both"):  suffixes.append("_7")
    if mode == "both":        suffixes = [""]  # both mode uses no suffix

    for suffix in suffixes:
        for sid in range(n_shards):
            p = out_dir / f"results_shard{sid}{suffix}.json"
            if not p.exists(): logger.warning("Missing %s", p.name); continue
            d = json.loads(p.read_text())
            if meta is None: meta = d.get("meta", {})
            s19.extend(d.get("_per_sample_segmentation", []))
            s7.extend(d.get("_per_sample_segmentation_7cat", []))
    agg = _agg_global if miou == "global" else _agg_per_image
    res = {}
    if s19:
        m, p = agg(s19, len(CLASSES), CLASSES, f"Merged {n_shards} shards — 19-class")
        res["segmentation"] = {"miou": m, "pixel_accuracy": p}
    if s7:
        m, p = agg(s7, 7, CAT7_NAMES, f"Merged {n_shards} shards — 7-cat")
        res["segmentation_7cat"] = {"miou": m, "pixel_accuracy": p}
    if not res: logger.error("No shard results!"); return
    if meta: meta["n_samples"] = max(len(s19),len(s7)); meta["n_shards"] = n_shards; res["meta"] = meta
    suffix = f"_{mode}" if mode != "both" else ""
    out = out_dir / f"results{suffix}.json"; out.write_text(json.dumps(res, indent=2))
    logger.info("Saved → %s", out)

# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="qwen",
                    choices=["qwen", "firered", "longcat", "omnigen2",
                             "kontext", "hunyuan", "generic"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", "-o", default="output/eval_Cityscapes")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="Inference steps (default: per-backend)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="True CFG scale (default: per-backend)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_id", type=int, default=None)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--mode", default="both", choices=["19","7","both"])
    ap.add_argument("--miou", default="global", choices=["global", "per_image"],
                    help="mIoU aggregation: 'global' (dataset-level, standard) or 'per_image' (legacy)")
    args = ap.parse_args()

    tag = Path(args.model).name
    out_dir = Path(args.output) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge(out_dir, args.n_shards, mode=args.mode, miou=args.miou); return

    pairs = _load_pairs(Path(args.data))
    logger.info("Cityscapes val: %d pairs", len(pairs))
    if not pairs: logger.error("No pairs found in %s", args.data); return

    sid = args.shard_id if args.shard_id is not None else 0
    indices = _shard(len(pairs), sid, args.n_shards, args.max_samples)
    logger.info("Shard %d/%d → %d samples, device=%s", sid, args.n_shards, len(indices), args.device)

    from vision_learner.pipeline import VisionLearner, DEFAULT_INFERENCE_PARAMS
    defaults = DEFAULT_INFERENCE_PARAMS.get(args.backend, {})
    steps = args.steps if args.steps is not None else defaults.get("num_inference_steps", 40)
    cfg = args.cfg if args.cfg is not None else defaults.get("true_cfg_scale", 4.0)
    logger.info("Inference params: steps=%d, cfg=%.1f (backend=%s)", steps, cfg, args.backend)
    learner = VisionLearner(model_name=args.model, backend=args.backend,
                            device=args.device,
                            num_inference_steps=steps, true_cfg_scale=cfg, seed=args.seed)

    t0 = time.time()
    results = evaluate(learner, pairs, indices, out_dir, mode=args.mode, miou=args.miou)
    elapsed = time.time() - t0

    results["meta"] = {"model":args.model, "backend":args.backend, "n_samples":len(indices),
                       "shard_id":sid, "n_shards":args.n_shards, "steps":steps,
                       "cfg":cfg, "seed":args.seed, "elapsed_sec":round(elapsed,1)}
    suffix = f"_{args.mode}" if args.mode != "both" else ""
    rp = out_dir / (f"results_shard{sid}{suffix}.json" if args.n_shards > 1 else f"results{suffix}.json")
    rp.write_text(json.dumps(results, indent=2))
    logger.info("Results saved → %s", rp)
    logger.info("Shard %d done: %.1f sec (%.1f sec/sample)", sid, elapsed, elapsed/max(len(indices),1))

if __name__ == "__main__":
    main()
