"""
VisionLearner — unified pipeline that turns image-editing models
into generalist vision models, following the Vision Banana paradigm.

Each vision task is reformulated as an image-editing request:
    (input_image, text_prompt)  ->  generated RGB image  ->  decoded output.

Supported backends:
    - qwen       : Qwen-Image-Edit-2511
    - firered    : FireRed-Image-Edit (Xiaohongshu)
    - longcat    : LongCat-Image-Edit (Meituan)
    - omnigen2   : OmniGen2 (VectorSpaceLab)
    - kontext    : Flux.1 Kontext [dev] (Black Forest Labs)
    - hunyuan    : HunyuanImage-3.0-Instruct (Tencent)
    - generic    : Any diffusers pipeline that accepts (image=, prompt=)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

from vision_learner.depth import DepthCodec
from vision_learner.normals import NormalCodec
from vision_learner.prompt_loader import (
    depth_prompt,
    normals_prompt,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Backend abstraction
# ======================================================================

class EditBackend(ABC):
    """Thin wrapper around a diffusers image-editing pipeline."""

    @abstractmethod
    def load(self, model_name: str, dtype: torch.dtype, device: str) -> None: ...

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, *,
                 num_inference_steps: int, guidance_scale: float,
                 true_cfg_scale: float, seed: int | None) -> Image.Image: ...


class QwenBackend(EditBackend):

    def __init__(self):
        self._pipe = None

    def load(self, model_name, dtype, device):
        from diffusers import QwenImageEditPlusPipeline

        logger.info("Loading Qwen backend: %s", model_name)
        self._pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self._pipe.to(device)
        self._pipe.set_progress_bar_config(disable=None)

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        gen = torch.manual_seed(seed) if seed is not None else None
        output = self._pipe(
            image=image,
            prompt=prompt,
            generator=gen,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        )
        return output.images[0]


class FireRedBackend(EditBackend):
    """FireRed-Image-Edit shares the same pipeline class as Qwen."""

    def __init__(self):
        self._pipe = None

    def load(self, model_name, dtype, device):
        from diffusers import QwenImageEditPlusPipeline

        logger.info("Loading FireRed backend: %s", model_name)
        self._pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self._pipe.to(device)
        self._pipe.set_progress_bar_config(disable=None)

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        gen = torch.manual_seed(seed) if seed is not None else None
        output = self._pipe(
            image=image,
            prompt=prompt,
            generator=gen,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        )
        return output.images[0]


class LongCatBackend(EditBackend):

    def __init__(self):
        self._pipe = None

    def load(self, model_name, dtype, device):
        from diffusers import LongCatImageEditPipeline

        logger.info("Loading LongCat backend: %s", model_name)
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self._pipe.to(device)
        self._pipe.set_progress_bar_config(disable=None)

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        gen = torch.manual_seed(seed) if seed is not None else None
        output = self._pipe(
            image,
            prompt,
            negative_prompt="",
            guidance_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=gen,
        )
        return output.images[0]


class OmniGen2Backend(EditBackend):
    """OmniGen2 — unified multimodal generation model (VectorSpaceLab).

    Requires the ``omnigen2`` package:
        pip install git+https://github.com/VectorSpaceLab/OmniGen2.git
    """

    def __init__(self):
        self._pipe = None
        self._device = "cuda"

    def load(self, model_name, dtype, device):
        try:
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.models.transformers.transformer_omnigen2 import (
                OmniGen2Transformer2DModel,
            )
        except ImportError:
            raise ImportError(
                "OmniGen2 backend requires the omnigen2 package. "
                "Install via: pip install git+https://github.com/VectorSpaceLab/OmniGen2.git"
            ) from None

        logger.info("Loading OmniGen2 backend: %s", model_name)
        self._pipe = OmniGen2Pipeline.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
        self._pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=dtype,
        )
        self._pipe.to(device)
        self._device = device

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        gen = (torch.Generator(device=self._device).manual_seed(seed)
               if seed is not None else None)
        full_prompt = f"<img><|image_1|></img> {prompt}"
        output = self._pipe(
            prompt=full_prompt,
            input_images=[image],
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=true_cfg_scale,
            image_guidance_scale=1.5,
            negative_prompt="",
            num_images_per_prompt=1,
            generator=gen,
            output_type="pil",
        )
        return output.images[0]


class FluxKontextBackend(EditBackend):
    """Flux.1 Kontext [dev] — instruction-based image editing (Black Forest Labs)."""

    def __init__(self):
        self._pipe = None

    def load(self, model_name, dtype, device):
        from diffusers import FluxKontextPipeline

        logger.info("Loading Flux Kontext backend: %s", model_name)
        self._pipe = FluxKontextPipeline.from_pretrained(
            model_name, torch_dtype=dtype,
        )
        self._pipe.to(device)
        self._pipe.set_progress_bar_config(disable=None)

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        output = self._pipe(
            image=image,
            prompt=prompt,
            guidance_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            generator=gen,
        )
        return output.images[0]


class HunyuanImageBackend(EditBackend):
    """HunyuanImage-3.0-Instruct — autoregressive MoE image editing (Tencent).

    Uses ``transformers.AutoModelForCausalLM`` with ``trust_remote_code``.
    The model is placed via ``device_map="auto"`` (the *device* arg is ignored).
    Recommended VRAM: >= 8 x 80 GB.
    """

    def __init__(self):
        self._model = None

    def load(self, model_name, dtype, device):
        from transformers import AutoModelForCausalLM

        logger.info("Loading HunyuanImage backend: %s", model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            moe_impl="eager",
            moe_drop_tokens=True,
        )
        if not hasattr(self._model.config, "model_version"):
            self._model.config.model_version = "v3"
        self._model.load_tokenizer(model_name)
        logger.info("HunyuanImage model and tokenizer loaded")

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name
        try:
            _, samples = self._model.generate_image(
                prompt=prompt,
                image=[tmp_path],
                seed=seed if seed is not None else 42,
                image_size="auto",
                use_system_prompt=None,
                bot_task="image",
                infer_align_image_size=True,
                diff_infer_steps=num_inference_steps,
                verbose=0,
            )
            return samples[0]
        finally:
            os.unlink(tmp_path)


class GenericBackend(EditBackend):
    """Fallback: tries AutoPipeline, passes image= and prompt=."""

    def __init__(self):
        self._pipe = None

    def load(self, model_name, dtype, device):
        from diffusers import AutoPipelineForImage2Image

        logger.info("Loading generic backend: %s", model_name)
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self._pipe.to(device)
        self._pipe.set_progress_bar_config(disable=None)

    def generate(self, image, prompt, *, num_inference_steps, guidance_scale,
                 true_cfg_scale, seed):
        gen = torch.manual_seed(seed) if seed is not None else None
        output = self._pipe(
            image=image,
            prompt=prompt,
            generator=gen,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        )
        return output.images[0]


# ======================================================================
# Backend registry
# ======================================================================

BACKENDS: dict[str, type[EditBackend]] = {
    "qwen": QwenBackend,
    "firered": FireRedBackend,
    "longcat": LongCatBackend,
    "omnigen2": OmniGen2Backend,
    "kontext": FluxKontextBackend,
    "hunyuan": HunyuanImageBackend,
    "generic": GenericBackend,
}

DEFAULT_MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen-Image-Edit-2511",
    "firered": "FireRedTeam/FireRed-Image-Edit-1.0",
    "longcat": "meituan-longcat/LongCat-Image-Edit",
    "omnigen2": "OmniGen2/OmniGen2",
    "kontext": "black-forest-labs/FLUX.1-Kontext-dev",
    "hunyuan": "tencent/HunyuanImage-3.0-Instruct",
}

DEFAULT_INFERENCE_PARAMS: dict[str, dict] = {
    "qwen":     {"num_inference_steps": 40, "true_cfg_scale": 4.0},
    "firered":  {"num_inference_steps": 40, "true_cfg_scale": 4.0},
    "longcat":  {"num_inference_steps": 50, "true_cfg_scale": 4.5},
    "omnigen2": {"num_inference_steps": 50, "true_cfg_scale": 5.0},
    "kontext":  {"num_inference_steps": 28, "true_cfg_scale": 2.5},
    "hunyuan":  {"num_inference_steps": 50, "true_cfg_scale": 1.0},
}


# ======================================================================
# VisionLearner — model-agnostic vision task interface
# ======================================================================

@dataclass
class VisionLearner:
    """
    Turns any image-editing model into a generalist vision model.

    Supported backends: qwen, firered, longcat, omnigen2, kontext, hunyuan, generic.
    """

    model_name: str = "Qwen/Qwen-Image-Edit-2511"
    backend: str = "qwen"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: Optional[int] = 0

    _backend_impl: EditBackend = field(default=None, init=False, repr=False)
    _depth_codec: DepthCodec = field(default_factory=DepthCodec, init=False)
    _normal_codec: NormalCodec = field(default_factory=NormalCodec, init=False)

    def load_model(self) -> None:
        if self._backend_impl is not None:
            return
        backend_cls = BACKENDS.get(self.backend)
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Available: {list(BACKENDS.keys())}"
            )
        self._backend_impl = backend_cls()
        self._backend_impl.load(self.model_name, self.dtype, self.device)
        logger.info("Model loaded on %s (backend=%s)", self.device, self.backend)

    def _generate(self, image: Image.Image, prompt: str) -> Image.Image:
        self.load_model()
        return self._backend_impl.generate(
            image, prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            true_cfg_scale=self.true_cfg_scale,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------
    def estimate_depth(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, Image.Image]:
        """
        Estimate metric depth from a single image.

        Returns:
            depth_map: (H, W) float array of metric depth in metres.
            vis_image: The raw generated false-colour depth visualisation.
        """
        prompt = depth_prompt()
        vis = self._generate(image, prompt)
        vis_resized = vis.resize(image.size, Image.BILINEAR)
        depth_map = self._depth_codec.decode_from_image(vis_resized)
        return depth_map, vis

    # ------------------------------------------------------------------
    # Surface normal estimation
    # ------------------------------------------------------------------
    def estimate_normals(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, Image.Image]:
        """
        Estimate surface normals from a single image.

        Returns:
            normals: (H, W, 3) float array of unit normal vectors.
            vis_image: The raw generated normal map image.
        """
        prompt = normals_prompt()
        vis = self._generate(image, prompt)
        vis_resized = vis.resize(image.size, Image.BILINEAR)
        normals = self._normal_codec.decode_from_image(vis_resized)
        return normals, vis

    # ------------------------------------------------------------------
    # Generic / custom task
    # ------------------------------------------------------------------
    def generate_raw(
        self, image: Image.Image, prompt: str
    ) -> Image.Image:
        """Run arbitrary prompt -- for experimentation."""
        return self._generate(image, prompt)
