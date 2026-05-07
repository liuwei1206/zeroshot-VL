"""
Vision Learner — Generalist vision tasks via image generation models.

Inspired by "Image Generators are Generalist Vision Learners" (Vision Banana),
this package reformulates classic vision tasks (depth, normals, segmentation)
as image-generation problems using any image-editing model.
"""

from vision_learner.depth import DepthCodec
from vision_learner.normals import NormalCodec
from vision_learner.pipeline import VisionLearner, BACKENDS, DEFAULT_MODELS
from vision_learner.prompt_loader import (
    depth_prompt,
    normals_prompt,
    render_prompt,
)
from vision_learner.segmentation import SegmentationDecoder

__all__ = [
    "DepthCodec",
    "NormalCodec",
    "SegmentationDecoder",
    "VisionLearner",
    "BACKENDS",
    "DEFAULT_MODELS",
    "render_prompt",
    "depth_prompt",
    "normals_prompt",
]
