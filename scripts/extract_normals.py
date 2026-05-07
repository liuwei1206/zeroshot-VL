"""Extract Ladicky normals from MarrRevisited .mat files into existing NYUv2 directory.

Only writes normals/ and normals_mask/, does NOT touch images/ or depth/.

Usage:
    python scripts/extract_normals.py --marr data/marr --nyuv2 data/nyuv2 --splits data/splits.mat
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marr", required=True, help="Path to directory with nm_*.mat files")
    parser.add_argument("--nyuv2", required=True, help="Path to existing data/nyuv2 directory")
    parser.add_argument("--splits", required=True, help="Path to splits.mat")
    args = parser.parse_args()

    marr_dir = Path(args.marr)
    out = Path(args.nyuv2)
    splits_path = Path(args.splits)

    splits = loadmat(str(splits_path))
    test_indices = (splits["testNdxs"].flatten() - 1).tolist()
    print(f"Eigen test split: {len(test_indices)} samples")

    (out / "normals").mkdir(parents=True, exist_ok=True)
    (out / "normals_mask").mkdir(parents=True, exist_ok=True)

    for out_idx, mat_idx in enumerate(tqdm(test_indices, desc="Extracting normals")):
        marr_path = marr_dir / f"nm_{mat_idx + 1:06d}.mat"
        if not marr_path.exists():
            print(f"  WARNING: {marr_path} not found, skipping")
            continue

        nm = loadmat(str(marr_path))
        normal = np.stack([nm["nx"], nm["ny"], nm["nz"]], axis=0).astype(np.float32)
        mask_bool = nm["depthValid"].astype(bool)

        np.save(out / "normals" / f"{out_idx:04d}.npy", normal)
        np.save(out / "normals_mask" / f"{out_idx:04d}.npy", mask_bool)

    print(f"Done → {out / 'normals'} ({len(test_indices)} samples, 480×640)")


if __name__ == "__main__":
    main()
