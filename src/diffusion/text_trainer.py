from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.text_conditioning import (
    drop_text_condition,
)


def compute_text_diffusion_loss(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    if x_0.ndim != 4:
        raise ValueError(
            "x_0 must have shape [B, C, H, W]"
        )

    batch_size = x_0.shape[0]

    if timesteps.shape != (
        batch_size,
    ):
        raise ValueError(
            "timesteps must have shape [B]"
        )

    if noise.shape != x_0.shape:
        raise ValueError(
            "noise must have same shape as x_0"
        )

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape [B, L]"
        )

    if (
        token_ids.shape[0]
        != batch_size
    ):
        raise ValueError(
            "token batch size must match x_0"
        )

    if (
        padding_mask.shape
        != token_ids.shape
    ):
        raise ValueError(
            "padding_mask must match token_ids"
        )

    if padding_mask.dtype != torch.bool:
        raise TypeError(
            "padding_mask must be bool"
        )

    timesteps = timesteps.to(
        device=x_0.device,
        dtype=torch.long,
    )

    noise = noise.to(
        device=x_0.device,
        dtype=x_0.dtype,
    )

    token_ids = token_ids.to(
        device=x_0.device,
        dtype=torch.long,
    )

    padding_mask = padding_mask.to(
        device=x_0.device,
        dtype=torch.bool,
    )

    x_t = scheduler.q_sample(
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    predicted_noise = model(
        x_t,
        timesteps,
        token_ids,
        padding_mask,
    )

    if (
        predicted_noise.shape
        != noise.shape
    ):
        raise ValueError(
            "model output shape must "
            "match noise shape"
        )

    return F.mse_loss(
        predicted_noise,
        noise,
    )


def text_training_step(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    prompt_dropout: float,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    if x_0.ndim != 4:
        raise ValueError(
            "x_0 must have shape [B,C,H,W]"
        )

    batch_size = x_0.shape[0]

    if batch_size <= 0:
        raise ValueError(
            "batch must not be empty"
        )

    token_ids = token_ids.to(
        device=x_0.device,
        dtype=torch.long,
    )

    padding_mask = padding_mask.to(
        device=x_0.device,
        dtype=torch.bool,
    )

    # ---------------------------------
    # Text classifier-free dropout
    # ---------------------------------

    (
        model_token_ids,
        model_padding_mask,
        dropped,
    ) = drop_text_condition(
        token_ids=token_ids,
        padding_mask=padding_mask,
        probability=prompt_dropout,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        generator=generator,
    )

    # ---------------------------------
    # Random t for each sample
    # ---------------------------------

    timesteps = torch.randint(
        low=0,
        high=scheduler.num_timesteps,
        size=(batch_size,),
        device=x_0.device,
        dtype=torch.long,
        generator=generator,
    )

    # ---------------------------------
    # epsilon ~ N(0, I)
    # ---------------------------------

    noise = torch.randn(
        x_0.shape,
        device=x_0.device,
        dtype=x_0.dtype,
        generator=generator,
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    loss = compute_text_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
        token_ids=model_token_ids,
        padding_mask=model_padding_mask,
    )

    if not torch.isfinite(
        loss
    ):
        raise FloatingPointError(
            "text diffusion loss "
            "is NaN or Inf"
        )

    loss.backward()

    for name, parameter in (
        model.named_parameters()
    ):
        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():
            raise FloatingPointError(
                f"non-finite gradient "
                f"in {name}"
            )

    optimizer.step()

    dropped_count = int(
        dropped.sum().item()
    )

    return {
        "loss": float(
            loss.detach().item()
        ),
        "batch_size": (
            batch_size
        ),
        "timestep_min": int(
            timesteps.min().item()
        ),
        "timestep_max": int(
            timesteps.max().item()
        ),
        "dropped_count": (
            dropped_count
        ),
        "dropped_fraction": (
            dropped_count
            / batch_size
        ),
    }


def train_text_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prompt_dropout: float,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_dropped = 0

    for batch in dataloader:
        x_0 = batch[
            "image"
        ].to(
            device=device,
            dtype=torch.float32,
        )

        token_ids = batch[
            "token_ids"
        ].to(
            device=device,
            dtype=torch.long,
        )

        padding_mask = batch[
            "padding_mask"
        ].to(
            device=device,
            dtype=torch.bool,
        )

        metrics = text_training_step(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            token_ids=token_ids,
            padding_mask=padding_mask,
            optimizer=optimizer,
            prompt_dropout=prompt_dropout,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            generator=generator,
        )

        batch_size = x_0.shape[0]

        total_loss += (
            metrics["loss"]
            * batch_size
        )

        total_samples += batch_size

        total_dropped += int(
            metrics[
                "dropped_count"
            ]
        )

    if total_samples == 0:
        raise ValueError(
            "training dataloader is empty"
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "num_samples": (
            total_samples
        ),
        "dropped_fraction": (
            total_dropped
            / total_samples
        ),
    }


@torch.no_grad()
def validate_text_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    """Validation uses full real prompts.

    No prompt dropout is applied.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x_0 = batch[
            "image"
        ].to(
            device=device,
            dtype=torch.float32,
        )

        token_ids = batch[
            "token_ids"
        ].to(
            device=device,
            dtype=torch.long,
        )

        padding_mask = batch[
            "padding_mask"
        ].to(
            device=device,
            dtype=torch.bool,
        )

        batch_size = x_0.shape[0]

        timesteps = torch.randint(
            low=0,
            high=scheduler.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

        noise = torch.randn(
            x_0.shape,
            device=device,
            dtype=x_0.dtype,
            generator=generator,
        )

        loss = compute_text_diffusion_loss(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
            token_ids=token_ids,
            padding_mask=padding_mask,
        )

        if not torch.isfinite(
            loss
        ):
            raise FloatingPointError(
                "validation loss is "
                "NaN or Inf"
            )

        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "validation dataloader is empty"
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "num_samples": (
            total_samples
        ),
    }