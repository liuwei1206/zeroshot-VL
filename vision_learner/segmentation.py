"""
Segmentation mask decoding from generated RGB images.

Semantic segmentation: the prompt specifies a class -> colour mapping.
Each pixel is assigned to the nearest specified colour in RGB space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Predefined colour palette — used when we need to auto-assign colours.
# ---------------------------------------------------------------------------
PALETTE = [
    ("red",       (255,   0,   0)),
    ("green",     (  0, 255,   0)),
    ("blue",      (  0,   0, 255)),
    ("yellow",    (255, 255,   0)),
    ("cyan",      (  0, 255, 255)),
    ("magenta",   (255,   0, 255)),
    ("orange",    (255, 165,   0)),
    ("purple",    (128,   0, 128)),
    ("pink",      (255, 192, 203)),
    ("lime",      (  0, 255, 128)),
    ("teal",      (  0, 128, 128)),
    ("brown",     (139,  69,  19)),
    ("navy",      (  0,   0, 128)),
    ("olive",     (128, 128,   0)),
    ("coral",     (255, 127,  80)),
    ("salmon",    (250, 128, 114)),
    ("gold",      (255, 215,   0)),
    ("violet",    (238, 130, 238)),
    ("turquoise", ( 64, 224, 208)),
    ("maroon",    (128,   0,   0)),
]


def _colour_rgb(idx: int) -> tuple[int, int, int]:
    return PALETTE[idx % len(PALETTE)][1]


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------

RGB = tuple[int, int, int]


@dataclass
class SemanticDecodeResult:
    masks: dict[str, np.ndarray]
    label_map: np.ndarray
    class_names: list[str]
    min_distance: np.ndarray
    unknown: np.ndarray


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SegmentationDecoder:
    """Decode generated RGB segmentation images into quantitative masks.

    Match each pixel to the nearest prompt-specified colour, optionally
    rejecting pixels farther than a threshold.
    """

    def __init__(self, colour_threshold: float = 45.0) -> None:
        self.colour_threshold = float(colour_threshold)

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    def decode_semantic(
        self,
        img: Image.Image,
        class_to_colour: dict[str, RGB],
        *,
        threshold: Optional[float] = None,
        force_assign: bool = False,
        include_unknown: bool = True,
        unknown_name: str = "unknown",
    ) -> SemanticDecodeResult:
        if not class_to_colour:
            raise ValueError("class_to_colour must not be empty")

        rgb = self._image_to_rgb(img).astype(np.float32)

        class_names = list(class_to_colour.keys())
        colours = np.asarray(
            [class_to_colour[name] for name in class_names],
            dtype=np.float32,
        )

        dists = np.linalg.norm(
            rgb[:, :, None, :] - colours[None, None, :, :],
            axis=-1,
        )

        nearest = np.argmin(dists, axis=-1).astype(np.int32)
        min_distance = np.min(dists, axis=-1)

        max_dist = self.colour_threshold if threshold is None else float(threshold)
        unknown = min_distance > max_dist

        label_map = nearest.copy()
        if not force_assign:
            label_map[unknown] = -1
        else:
            unknown = np.zeros_like(unknown, dtype=bool)

        masks = {
            name: label_map == idx
            for idx, name in enumerate(class_names)
        }

        if include_unknown and not force_assign:
            masks[unknown_name] = unknown

        return SemanticDecodeResult(
            masks=masks,
            label_map=label_map,
            class_names=class_names,
            min_distance=min_distance,
            unknown=unknown,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def overlay(
        image: Image.Image,
        masks: dict[str, np.ndarray],
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlay coloured masks on the original image."""
        base = np.asarray(image.convert("RGB"), dtype=np.float64)
        overlay_arr = base.copy()
        for i, (cls_name, mask) in enumerate(masks.items()):
            if cls_name in ("background", "unknown"):
                continue
            colour = np.array(_colour_rgb(i), dtype=np.float64)
            overlay_arr[mask] = overlay_arr[mask] * (1 - alpha) + colour * alpha
        return Image.fromarray(overlay_arr.clip(0, 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_rgb(img: Image.Image) -> np.ndarray:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
