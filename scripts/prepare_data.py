#!/usr/bin/env python3
"""
Download and prepare evaluation datasets.

Datasets by task:

  Semantic segmentation:  Cityscapes val
  Metric depth:           NYUv2, iBims1, ETH3D, DIODE, KITTI, nuScenes
  Surface normals:        NYUv2, DIODE, ScanNet, Virtual KITTI 2

Usage:
    python scripts/prepare_data.py --dataset nyuv2 --output data/
    python scripts/prepare_data.py --all --output data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def _skip_if_exists(out: Path) -> bool:
    if (out / "images").exists() and any((out / "images").iterdir()):
        print("  Already exists, skipping.")
        return True
    return False


# ===== Semantic Segmentation =====

def prepare_cityscapes(output: Path, max_samples: int | None = None) -> None:
    print("\n=== Cityscapes val ===")
    out = output / "cityscapes"
    if _skip_if_exists(out):
        return
    print(f"""
  Cityscapes requires registration. Download from:
    https://www.cityscapes-dataset.com/downloads/

  Files needed:
    - leftImg8bit_trainvaltest.zip  (images)
    - gtFine_trainvaltest.zip       (labels)

  Extract and organise as:
    {out}/leftImg8bit/val/...
    {out}/gtFine/val/...

  Or use:
    pip install cityscapesscripts
    csDownload -d {out} gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
""")


# ===== Metric Depth Estimation =====

# NYUv2 camera intrinsics (from the official toolbox)
_NYU_FX = 5.1885790117450188e+02
_NYU_FY = 5.1946961112127485e+02
_NYU_CX = 3.2558244941119034e+02
_NYU_CY = 2.5373616633400465e+02


def _depth_to_normals_nyuv2(
    depth: np.ndarray,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute surface normals from a depth map using cross-product of
    back-projected 3D point neighbours.

    Args:
        depth: (H, W) float32 metric depth in metres.
        window: half-window for selecting neighbour pixels (larger = smoother).

    Returns:
        normals: (3, H, W) float32, unit surface normals.
        valid:   (H, W) bool, pixels with reliable normals.
    """
    H, W = depth.shape
    valid = (depth > 1e-3) & np.isfinite(depth)

    v, u = np.meshgrid(np.arange(H, dtype=np.float32),
                        np.arange(W, dtype=np.float32), indexing="ij")
    z = depth.astype(np.float64)
    x = (u - _NYU_CX) * z / _NYU_FX
    y = (v - _NYU_CY) * z / _NYU_FY

    # Shifted neighbours (up/down/left/right by `window` pixels)
    def _shift(arr, dy, dx):
        out = np.zeros_like(arr)
        sy = slice(max(dy, 0), H + min(dy, 0))
        sx = slice(max(dx, 0), W + min(dx, 0))
        ty = slice(max(-dy, 0), H + min(-dy, 0))
        tx = slice(max(-dx, 0), W + min(-dx, 0))
        out[ty, tx] = arr[sy, sx]
        return out

    dxp = np.stack([_shift(x, 0, window) - x,
                     _shift(y, 0, window) - y,
                     _shift(z, 0, window) - z], axis=0)  # (3, H, W)
    dyp = np.stack([_shift(x, window, 0) - x,
                     _shift(y, window, 0) - y,
                     _shift(z, window, 0) - z], axis=0)

    normals = np.cross(dxp, dyp, axis=0)  # (3, H, W)
    norm = np.linalg.norm(normals, axis=0, keepdims=True) + 1e-10
    normals = normals / norm
    normals = normals.astype(np.float32)

    boundary = window
    valid_normals = valid.copy()
    valid_normals[:boundary, :] = False
    valid_normals[-boundary:, :] = False
    valid_normals[:, :boundary] = False
    valid_normals[:, -boundary:] = False

    return normals, valid_normals


def prepare_nyuv2(output: Path, max_samples: int | None = None) -> None:
    """Prepare NYUv2 val split from official sources.

    Required downloads (place in <output>/):
        1. nyu_depth_v2_labeled.mat  (2.8 GB)  — images + filled depth
           http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
        2. splits.mat                           — Eigen train/test split
           http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
        3. (optional) nyu_normals_gt/           — Ladicky et al. precomputed normals + masks
           If not present, normals are computed from depth via cross-product of
           back-projected 3D point neighbours using NYUv2 camera intrinsics.

    Output format:
        images/NNNN.png        — RGB, 480×640
        depth/NNNN.npy         — float32 (H, W), metric depth in metres (filled)
        normals/NNNN.npy       — float32 (3, H, W), unit surface normals
        normals_mask/NNNN.npy  — bool (H, W), valid region for normals
    """
    print("\n=== NYU Depth V2 (depth + normals) ===")
    out = output / "nyuv2"
    if _skip_if_exists(out):
        return

    import h5py
    from scipy.io import loadmat

    mat_path = output / "nyu_depth_v2_labeled.mat"
    splits_path = output / "splits.mat"
    normals_dir = output / "nyu_normals_gt"

    missing = []
    if not mat_path.exists():
        missing.append(f"  wget -O {mat_path} http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    if not splits_path.exists():
        missing.append(f"  wget -O {splits_path} http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat")
    if missing:
        print("\n  Missing files. Please download first:\n")
        for m in missing:
            print(m)
        print()
        return

    # Normals source detection (priority: MarrRevisited .mat > Ladicky PNG > compute from depth)
    marr_dir = output / "marr"
    use_marr = marr_dir.exists() and any(marr_dir.glob("nm_*.mat"))
    use_ladicky_png = (not use_marr) and normals_dir.exists() and (normals_dir / "normals").exists()

    if use_marr:
        marr_files = sorted(marr_dir.glob("nm_*.mat"))
        print(f"  Using MarrRevisited / Ladicky normals (.mat): {len(marr_files)} files")
    elif use_ladicky_png:
        print("  Using Ladicky et al. precomputed normals (PNG)")
    else:
        print("  No external normals GT found — computing normals from filled depth")

    # Load Eigen test split (654 indices, 0-indexed)
    splits = loadmat(str(splits_path))
    test_indices = (splits["testNdxs"].flatten() - 1).tolist()
    print(f"  Eigen test split: {len(test_indices)} samples")

    if max_samples:
        test_indices = test_indices[:max_samples]

    if use_ladicky_png:
        normals_img_dir = normals_dir / "normals"
        masks_img_dir = normals_dir / "masks"
        normals_files = sorted(normals_img_dir.glob("*.png"))
        masks_files = sorted(masks_img_dir.glob("*.png"))
        print(f"  Found {len(normals_files)} normal maps, {len(masks_files)} masks")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "normals").mkdir(parents=True, exist_ok=True)
    (out / "normals_mask").mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {len(test_indices)} test samples ...")
    with h5py.File(str(mat_path), "r") as f:
        for out_idx, mat_idx in enumerate(tqdm(test_indices, desc="NYUv2")):
            img = np.array(f["images"][mat_idx], dtype=np.uint8)    # (3, 640, 480)
            img = np.transpose(img, (2, 1, 0))                      # (480, 640, 3)
            Image.fromarray(img).save(out / "images" / f"{out_idx:04d}.png")

            depth = np.array(f["depths"][mat_idx], dtype=np.float32)  # (640, 480)
            depth = depth.T                                            # (480, 640)
            np.save(out / "depth" / f"{out_idx:04d}.npy", depth)

            if use_marr:
                # MarrRevisited .mat: 1-indexed filenames (nm_000001.mat = NYUv2 image #1)
                marr_path = marr_dir / f"nm_{mat_idx + 1:06d}.mat"
                nm = loadmat(str(marr_path))
                normal = np.stack([nm["nx"], nm["ny"], nm["nz"]], axis=0).astype(np.float32)  # (3, 480, 640)
                mask_bool = nm["depthValid"].astype(bool)  # (480, 640)
                np.save(out / "normals" / f"{out_idx:04d}.npy", normal)
                np.save(out / "normals_mask" / f"{out_idx:04d}.npy", mask_bool)
            elif use_ladicky_png:
                normal_img = np.array(Image.open(normals_files[mat_idx]))
                normal = normal_img.astype(np.float32) / 255.0 * 2.0 - 1.0
                normal = np.transpose(normal, (2, 0, 1))               # (3, H, W)
                np.save(out / "normals" / f"{out_idx:04d}.npy", normal)

                mask = np.array(Image.open(masks_files[mat_idx]))
                mask_bool = mask > 0
                np.save(out / "normals_mask" / f"{out_idx:04d}.npy", mask_bool)
            else:
                normal, mask_bool = _depth_to_normals_nyuv2(depth)
                np.save(out / "normals" / f"{out_idx:04d}.npy", normal)
                np.save(out / "normals_mask" / f"{out_idx:04d}.npy", mask_bool)

    src_label = "MarrRevisited/Ladicky" if use_marr else ("Ladicky PNG" if use_ladicky_png else "computed from depth")
    print(f"  Done → {out}  ({len(test_indices)} samples)")
    print(f"    images:       {out / 'images'}       (PNG, 480×640)")
    print(f"    depth:        {out / 'depth'}        (.npy float32, H×W, metres, filled)")
    print(f"    normals:      {out / 'normals'}      (.npy float32, 3×H×W, {src_label})")
    print(f"    normals_mask: {out / 'normals_mask'} (.npy bool, H×W, valid region)")


def prepare_ibims1(output: Path, max_samples: int | None = None) -> None:
    print("\n=== iBims-1 ===")
    out = output / "ibims1"
    if _skip_if_exists(out):
        return
    print(f"""
  iBims-1 (Independently acquired Binocular Matching Stereo images).
  Download from:
    https://www.bgu.tum.de/lmf/ibims1/

  Organise as:
    {out}/images/   and   {out}/depth/
""")


def prepare_eth3d(output: Path, max_samples: int | None = None) -> None:
    print("\n=== ETH3D ===")
    out = output / "eth3d"
    if _skip_if_exists(out):
        return
    print(f"""
  ETH3D dataset for multi-view stereo and depth evaluation.
  Download from:
    https://www.eth3d.net/datasets

  Organise as:
    {out}/images/   and   {out}/depth/
""")


def prepare_diode(output: Path, max_samples: int | None = None) -> None:
    """Convert local raw DIODE val data to our standard format.

    Expects raw DIODE structure at <output>/DIODE/:
        DIODE/indoors/scene_XXXXX/scan_YYYYY/XXXXX_YYYYY_indoors_ZZZ_WWW.png
        DIODE/indoors/scene_XXXXX/scan_YYYYY/XXXXX_YYYYY_indoors_ZZZ_WWW_depth.npy
        DIODE/indoors/scene_XXXXX/scan_YYYYY/XXXXX_YYYYY_indoors_ZZZ_WWW_depth_mask.npy
        DIODE/outdoor/scene_XXXXX/scan_YYYYY/...

    Produces:
        diode_indoor/images/NNNN.png  diode_indoor/depth/NNNN.npy  diode_indoor/depth_mask/NNNN.npy
        diode_outdoor/images/NNNN.png diode_outdoor/depth/NNNN.npy diode_outdoor/depth_mask/NNNN.npy
    """
    raw_root = output / "DIODE"
    if not raw_root.exists():
        print("\n=== DIODE ===")
        print(f"""
  Raw DIODE data not found at {raw_root}.
  Download val split (no registration needed):
    wget http://diode-dataset.s3.amazonaws.com/val.tar.gz
    tar xf val.tar.gz -C {output}
  This creates {raw_root}/indoors/ and {raw_root}/outdoor/.
""")
        return

    for subset, dirname in [("indoors", "diode_indoor"), ("outdoor", "diode_outdoor")]:
        print(f"\n=== DIODE {subset} ===")
        raw_dir = raw_root / subset
        if not raw_dir.exists():
            print(f"  {raw_dir} not found, skipping.")
            continue

        out = output / dirname
        if _skip_if_exists(out):
            continue

        imgs = sorted(raw_dir.rglob("*.png"))
        imgs = [p for p in imgs if "_depth" not in p.name and "_normal" not in p.name]
        if max_samples:
            imgs = imgs[:max_samples]

        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "depth").mkdir(parents=True, exist_ok=True)
        (out / "depth_mask").mkdir(parents=True, exist_ok=True)

        converted = 0
        for i, img_path in enumerate(tqdm(imgs, desc=f"DIODE {subset}")):
            stem = img_path.stem
            depth_path = img_path.parent / f"{stem}_depth.npy"
            mask_path = img_path.parent / f"{stem}_depth_mask.npy"

            if not depth_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            img.save(out / "images" / f"{i:04d}.png")

            depth = np.load(depth_path).squeeze().astype(np.float32)
            np.save(out / "depth" / f"{i:04d}.npy", depth)

            if mask_path.exists():
                mask = np.load(mask_path).squeeze()
                np.save(out / "depth_mask" / f"{i:04d}.npy", mask)

            converted += 1

        print(f"  Done → {out}  ({converted} samples)")
        print(f"    images:     {out / 'images'}  (PNG)")
        print(f"    depth:      {out / 'depth'}   (.npy float32)")
        print(f"    depth_mask: {out / 'depth_mask'} (.npy validity mask)")


def prepare_kitti(output: Path, max_samples: int | None = None) -> None:
    print("\n=== KITTI Depth ===")
    out = output / "kitti"
    if _skip_if_exists(out):
        return
    print(f"""
  KITTI depth evaluation benchmark. Download from:
    https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

  Files needed:
    - data_depth_annotated.zip (ground truth)
    - Download raw images from the corresponding drives

  Organise as:
    {out}/images/   and   {out}/depth/
""")


def prepare_nuscenes(output: Path, max_samples: int | None = None) -> None:
    print("\n=== nuScenes ===")
    out = output / "nuscenes"
    if _skip_if_exists(out):
        return
    print(f"""
  nuScenes dataset. Download from:
    https://www.nuscenes.org/nuscenes#download

  Requires registration. Download the "Full dataset (v1.0)" mini or trainval split.

  Organise as:
    {out}/images/   and   {out}/depth/
""")


# ===== Surface Normal Estimation =====
# NYUv2 and DIODE-Indoor are shared with depth (above).

def prepare_scannet(output: Path, max_samples: int | None = None) -> None:
    print("\n=== ScanNet ===")
    out = output / "scannet"
    if _skip_if_exists(out):
        return
    print(f"""
  ScanNet dataset. Requires signing a terms-of-use agreement.
  Apply at:
    http://www.scan-net.org/

  After approval, download with the provided script:
    python download-scannet.py -o {out} --type .sens

  Organise as:
    {out}/images/   and   {out}/normals/
""")


def prepare_vkitti2(output: Path, max_samples: int | None = None) -> None:
    print("\n=== Virtual KITTI 2 ===")
    out = output / "vkitti2"
    if _skip_if_exists(out):
        return
    print(f"""
  Download from:
    https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

    mkdir -p {out} && cd {out}
    wget .../vkitti_2.0.3_rgb.tar && tar xf vkitti_2.0.3_rgb.tar
    wget .../vkitti_2.0.3_depth.tar && tar xf vkitti_2.0.3_depth.tar
    wget .../vkitti_2.0.3_forwardFlow.tar  (for normals if available)

  Depth is uint16 PNG in centimetres.
""")


# ===== Main =====

DATASETS = {
    "cityscapes": prepare_cityscapes,
    "nyuv2": prepare_nyuv2,
    "ibims1": prepare_ibims1,
    "eth3d": prepare_eth3d,
    "diode": prepare_diode,
    "kitti": prepare_kitti,
    "nuscenes": prepare_nuscenes,
    # Surface normal estimation (NYUv2 & DIODE shared with depth)
    "scannet": prepare_scannet,
    "vkitti2": prepare_vkitti2,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download/prepare Vision Banana evaluation datasets.",
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Which dataset to prepare")
    parser.add_argument("--all", action="store_true",
                        help="Prepare all datasets")
    parser.add_argument("--output", "-o", default="data",
                        help="Output root directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    args = parser.parse_args()

    output = Path(args.output)

    if args.all:
        for name, fn in DATASETS.items():
            try:
                fn(output, max_samples=args.max_samples)
            except Exception as e:
                print(f"  ERROR on {name}: {e}")
                print("  Skipping, continuing with next dataset...")
    elif args.dataset:
        DATASETS[args.dataset](output, max_samples=args.max_samples)
    else:
        parser.print_help()
        print("\nAvailable datasets:", ", ".join(DATASETS.keys()))


if __name__ == "__main__":
    main()
