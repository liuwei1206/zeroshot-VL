"""
Surface normal encoding / decoding following the Vision Banana paper.

Surface normals are unit vectors (x, y, z) ∈ [-1, 1]³  in camera space
using a right-handed coordinate system (+x right, +y up, +z out of image).

The mapping to RGB is the standard normal-map convention:
    R = (x + 1) / 2
    G = (y + 1) / 2
    B = (z + 1) / 2

Characteristic colours:
    Facing left  (-1, 0, 0) → pinkish-red  (0.0, 0.5, 0.5)
    Facing up    ( 0, 1, 0) → light green   (0.5, 1.0, 0.5)
    Facing camera( 0, 0, 1) → light blue    (0.5, 0.5, 1.0)
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class NormalCodec:
    """Bijective mapping between surface normals and RGB colours."""

    @staticmethod
    def encode(normals: np.ndarray) -> np.ndarray:
        """(H, W, 3) unit normals → (H, W, 3) RGB in [0, 1]."""
        return (normals + 1.0) / 2.0

    @staticmethod
    def decode(rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) RGB in [0, 1] → (H, W, 3) unit normals (re-normalised)."""
        normals = rgb * 2.0 - 1.0
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return normals / norms

    @staticmethod
    def encode_to_image(normals: np.ndarray) -> Image.Image:
        rgb = NormalCodec.encode(normals)
        return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))

    @staticmethod
    def decode_from_image(img: Image.Image) -> np.ndarray:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        return NormalCodec.decode(rgb)
