"""Inference adapters used by the local portfolio playground.

This module owns checkpoint restoration, input preprocessing, and calls into
the from-scratch models under ``src/``. It deliberately contains no
Streamlit code so its contracts can be tested without starting a web server.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import hashlib
import os
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from src.captioning.generation import beam_generate, greedy_generate
from src.captioning.model import CaptionModel
from src.data.product_preprocessing import letterbox_square
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.text_conditional_unet import TextConditionalUNet
from src.diffusion.text_sampler import sample_ddpm_text_cfg
from src.text.tokenizer import decode, encode, padding_mask, tokenize
from src.text.vocabulary import SPECIAL_TOKENS, Vocabulary


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTION_CKPT_PATH = REPOSITORY_ROOT / "checkpoints" / "image_captioning.pt"
DIFFUSION_CKPT_PATH = (
    REPOSITORY_ROOT / "checkpoints" / "text_to_image_diffusion.pt"
)
DEMO_RELEASE_TAG = "demo-v1"
DEMO_RELEASE_BASE_URL = (
    "https://github.com/dangkhoi3107/multimodal-caption-diffusion/"
    f"releases/download/{DEMO_RELEASE_TAG}"
)
CAPTION_CKPT_URL = os.environ.get(
    "CAPTION_CKPT_URL",
    f"{DEMO_RELEASE_BASE_URL}/image_captioning.pt",
)
DIFFUSION_CKPT_URL = os.environ.get(
    "DIFFUSION_CKPT_URL",
    f"{DEMO_RELEASE_BASE_URL}/text_to_image_diffusion.pt",
)
CAPTION_CKPT_SHA256 = (
    "7858c160dc9d773fc50012a2305473702753ff264b625a890dde906f5e3bc09b"
)
DIFFUSION_CKPT_SHA256 = (
    "b9de99b913bdbce942afc107b3579b9dce326dfc17b47c3524c1c31d92e759db"
)

Checkpoint = dict[str, Any]
CaptionStrategy = Literal["greedy", "beam"]
DenoisingPreviewCallback = Callable[[int, int, int, Image.Image], None]


def ensure_checkpoint_available(
    path: Path,
    download_url: str,
    expected_sha256: str,
) -> Path:
    """Return a local checkpoint, downloading and verifying it when absent.

    Existing local files are never overwritten. A missing checkpoint is streamed
    to a uniquely named temporary file, checked against ``expected_sha256``, and
    atomically moved into place only after verification succeeds.
    """

    resolved_path = path.expanduser().resolve()
    if resolved_path.is_file():
        return resolved_path
    if len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_name(
        f".{resolved_path.name}.{uuid4().hex}.part"
    )
    digest = hashlib.sha256()
    request = Request(
        download_url,
        headers={"User-Agent": "multimodal-caption-diffusion-demo/1.0"},
    )

    try:
        with urlopen(request, timeout=180) as response, temporary_path.open(
            "wb"
        ) as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)

        actual_sha256 = digest.hexdigest()
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                "Downloaded checkpoint checksum mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        temporary_path.replace(resolved_path)
    except (OSError, URLError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download checkpoint from {download_url}: {error}"
        ) from error

    return resolved_path


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path) -> Checkpoint:
    """Load and validate one local project checkpoint on CPU.

    The checkpoint is restored with ``weights_only=True`` because the project
    stores tensors and plain Python containers only. Moving model tensors to
    the selected inference device happens after state restoration.
    """

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved_path}. "
            "See README.md#local-checkpoints for the expected files."
        )

    checkpoint = torch.load(
        resolved_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a dictionary")

    required_keys = {"model_state_dict", "config", "vocabulary"}
    missing_keys = sorted(required_keys.difference(checkpoint))
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            + ", ".join(missing_keys)
        )

    return checkpoint


def restore_vocabulary(data: Mapping[str, Any]) -> Vocabulary:
    """Restore and validate a vocabulary serialized inside a checkpoint."""

    try:
        token_to_id = {
            str(token): int(token_id)
            for token, token_id in data["token_to_id"].items()
        }
        id_to_token = tuple(str(token) for token in data["id_to_token"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid checkpoint vocabulary") from error

    vocabulary = Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )
    vocabulary.validate()
    return vocabulary


def preprocess_product_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """Convert a PIL image to a normalized model batch.

    Args:
        image: Arbitrary-size PIL image.
        image_size: Square model input resolution.

    Returns:
        FloatTensor ``[1, 3, image_size, image_size]`` in ``[-1, 1]``.

    Aspect ratio is preserved with the same white letterbox policy used by
    Phase 0. The function does not segment a shelf photo; the uploaded image
    should already contain one centered product.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.uint8).copy()
    square = letterbox_square(
        image=array,
        size=image_size,
        fill=(255, 255, 255),
    )

    tensor = torch.from_numpy(square).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0
    tensor = tensor.unsqueeze(0)

    expected_shape = (1, 3, image_size, image_size)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Expected preprocessed shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("preprocessed image contains non-finite values")

    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert one normalized ``[3,H,W]`` tensor in ``[-1,1]`` to RGB PIL."""

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("tensor must have shape [3, H, W]")

    image = tensor.detach().cpu().float().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().to(torch.uint8)
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


class CaptioningDemo:
    """Restore the Phase 4 model and generate one product caption."""

    def __init__(
        self,
        checkpoint_path: Path = CAPTION_CKPT_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.device = device or get_device()
        if self.checkpoint_path == CAPTION_CKPT_PATH.resolve():
            ensure_checkpoint_available(
                path=self.checkpoint_path,
                download_url=CAPTION_CKPT_URL,
                expected_sha256=CAPTION_CKPT_SHA256,
            )
        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.config = self.checkpoint["config"]
        self.vocabulary = restore_vocabulary(self.checkpoint["vocabulary"])

        self.model = self._build_model()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> CaptionModel:
        """Recreate the exact random-init architecture saved at training."""

        data_cfg = self.config["data"]
        text_cfg = self.config["text"]
        model_cfg = self.config["model"]
        image_cfg = model_cfg["image_encoder"]
        decoder_cfg = model_cfg["decoder"]

        return CaptionModel(
            vocab_size=len(self.vocabulary),
            pad_id=self.vocabulary.pad_id,
            image_size=int(data_cfg["image_size"]),
            in_channels=int(image_cfg["in_channels"]),
            base_channels=int(image_cfg["base_channels"]),
            model_dim=int(model_cfg["model_dim"]),
            max_length=int(text_cfg["sequence_length"]),
            num_heads=int(decoder_cfg["num_heads"]),
            num_layers=int(decoder_cfg["num_layers"]),
            feedforward_dim=int(decoder_cfg["feedforward_dim"]),
            dropout=float(decoder_cfg["dropout"]),
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of learned parameters."""

        return sum(parameter.numel() for parameter in self.model.parameters())

    def generate_caption(
        self,
        image: Image.Image,
        strategy: CaptionStrategy = "greedy",
        beam_size: int = 3,
        length_penalty: float = 0.6,
    ) -> str:
        """Generate a caption for one product crop.

        ``image`` becomes ``[1,3,64,64]`` in ``[-1,1]``. Greedy or beam
        decoding returns token IDs ``[1,L]`` which are decoded without
        ``PAD/BOS/EOS`` tokens.
        """

        if strategy not in ("greedy", "beam"):
            raise ValueError("strategy must be 'greedy' or 'beam'")

        image_size = int(self.config["data"]["image_size"])
        image_tensor = preprocess_product_image(image, image_size).to(self.device)
        max_length = int(self.config["generation"]["max_length"])

        generation_arguments = {
            "model": self.model,
            "images": image_tensor,
            "bos_id": self.vocabulary.bos_id,
            "eos_id": self.vocabulary.eos_id,
            "pad_id": self.vocabulary.pad_id,
            "max_length": max_length,
        }

        if strategy == "greedy":
            token_ids = greedy_generate(**generation_arguments)
        else:
            token_ids = beam_generate(
                **generation_arguments,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )

        caption = decode(token_ids[0], vocabulary=self.vocabulary)
        if not caption:
            return "(model emitted no content tokens)"
        return caption


class DiffusionDemo:
    """Restore the Phase 3 text-conditioned DDPM and sample one image."""

    def __init__(
        self,
        checkpoint_path: Path = DIFFUSION_CKPT_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.device = device or get_device()
        if self.checkpoint_path == DIFFUSION_CKPT_PATH.resolve():
            ensure_checkpoint_available(
                path=self.checkpoint_path,
                download_url=DIFFUSION_CKPT_URL,
                expected_sha256=DIFFUSION_CKPT_SHA256,
            )
        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.config = self.checkpoint["config"]
        self.vocabulary = restore_vocabulary(self.checkpoint["vocabulary"])

        self.model = self._build_model()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        diffusion_cfg = self.config["diffusion"]
        self.scheduler = DDPMScheduler(
            num_timesteps=int(diffusion_cfg["num_timesteps"]),
            beta_start=float(diffusion_cfg["beta_start"]),
            beta_end=float(diffusion_cfg["beta_end"]),
        )

    def _build_model(self) -> TextConditionalUNet:
        """Recreate the exact text encoder and conditional U-Net."""

        text_cfg = self.config["text"]
        model_cfg = self.config["model"]

        return TextConditionalUNet(
            vocab_size=len(self.vocabulary),
            pad_id=self.vocabulary.pad_id,
            max_length=int(text_cfg["max_length"]),
            text_embedding_dim=int(text_cfg["embedding_dim"]),
            text_num_heads=int(text_cfg["num_heads"]),
            text_num_layers=int(text_cfg["num_layers"]),
            text_feedforward_dim=int(text_cfg["feedforward_dim"]),
            text_dropout=float(text_cfg["dropout"]),
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            base_channels=int(model_cfg["base_channels"]),
            time_embedding_dim=int(model_cfg["time_embedding_dim"]),
            time_dim=int(model_cfg["time_dim"]),
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of learned parameters."""

        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def supported_tokens(self) -> tuple[str, ...]:
        """Return non-special prompt tokens learned from training captions."""

        return tuple(
            token
            for token in self.vocabulary.id_to_token
            if token not in SPECIAL_TOKENS
        )

    def unknown_prompt_tokens(self, prompt: str) -> tuple[str, ...]:
        """Return normalized prompt tokens mapped to ``UNK`` by the model."""

        return tuple(
            token
            for token in tokenize(prompt)
            if token not in self.vocabulary.token_to_id
        )

    def generate_image(
        self,
        prompt: str,
        guidance_scale: float = 2.0,
        seed: int = 42,
        progress_callback: DenoisingPreviewCallback | None = None,
        preview_interval: int = 50,
    ) -> Image.Image:
        """Sample one deterministic ``64x64`` RGB image from text.

        Prompt IDs have shape ``[1,L]`` and the reverse chain has shape
        ``[1,3,64,64]``. A local ``torch.Generator`` makes the result stable
        for the same prompt, guidance scale, seed, checkpoint, and device.
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not tokenize(prompt):
            raise ValueError("prompt must contain at least one word or number")
        if guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if preview_interval <= 0:
            raise ValueError("preview_interval must be positive")

        max_length = int(self.config["text"]["max_length"])
        token_ids = encode(
            text=prompt,
            vocabulary=self.vocabulary,
            max_length=max_length,
        ).unsqueeze(0)
        prompt_mask = padding_mask(token_ids, self.vocabulary)

        image_size = int(self.config["data"]["image_size"])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        def render_progress(
            completed_steps: int,
            total_steps: int,
            timestep: int,
            snapshot: torch.Tensor,
        ) -> None:
            """Convert one sampler snapshot into a UI-safe RGB preview."""

            if progress_callback is not None:
                progress_callback(
                    completed_steps,
                    total_steps,
                    timestep,
                    tensor_to_pil(snapshot[0]),
                )

        samples = sample_ddpm_text_cfg(
            model=self.model,
            scheduler=self.scheduler,
            token_ids=token_ids,
            padding_mask=prompt_mask,
            bos_id=self.vocabulary.bos_id,
            eos_id=self.vocabulary.eos_id,
            pad_id=self.vocabulary.pad_id,
            shape=(1, 3, image_size, image_size),
            device=self.device,
            guidance_scale=guidance_scale,
            generator=generator,
            progress_callback=(
                render_progress
                if progress_callback is not None
                else None
            ),
            progress_interval=preview_interval,
        )

        return tensor_to_pil(samples[0])


@lru_cache(maxsize=2)
def get_caption_demo(
    checkpoint_path: Path = CAPTION_CKPT_PATH,
) -> CaptioningDemo:
    """Return a process-local cached captioning inference adapter."""

    return CaptioningDemo(checkpoint_path=checkpoint_path)


@lru_cache(maxsize=2)
def get_diffusion_demo(
    checkpoint_path: Path = DIFFUSION_CKPT_PATH,
) -> DiffusionDemo:
    """Return a process-local cached diffusion inference adapter."""

    return DiffusionDemo(checkpoint_path=checkpoint_path)
