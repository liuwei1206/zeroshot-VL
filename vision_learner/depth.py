"""
Depth decoding via perceived luminance extraction (ITU-R BT.709).

The grayscale depth prompt instructs the model to produce a near-grayscale
image where brightness encodes relative depth.  The decoder extracts
perceived luminance from the RGB output as a depth proxy.

Quantitative evaluation uses affine-invariant alignment (scale + offset)
against ground truth, so the decoder only needs to preserve the *ordering*
of depth values, not their absolute scale.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_LUMA_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


class DepthCodec:
    """Decode model-generated depth visualizations via luminance extraction."""

    @staticmethod
    def decode(rgb: np.ndarray) -> np.ndarray:
        """RGB float array (H, W, 3) in [0, 1] → relative depth (H, W)."""
        return np.dot(rgb, _LUMA_W)

    @staticmethod
    def decode_from_image(img: Image.Image) -> np.ndarray:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        return np.dot(rgb, _LUMA_W)
