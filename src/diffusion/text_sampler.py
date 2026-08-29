from __future__ import annotations

from collections.abc import Callable

import torch

from src.diffusion.conditional_sampler import (
    classifier_free_guidance,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


SamplingProgressCallback = Callable[
    [int, int, int, torch.Tensor],
    None,
]


@torch.no_grad()
def sample_ddpm_text_cfg(
    model,
    scheduler: DDPMScheduler,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    shape: tuple[int, int, int, int],
    device: torch.device,
    guidance_scale: float = 2.0,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    trajectory_interval: int = 100,
    progress_callback: SamplingProgressCallback | None = None,
    progress_interval: int = 50,
):
    """Sample a text-conditioned DDPM chain with optional live snapshots.

    ``progress_callback`` receives ``(completed_steps, total_steps, timestep,
    snapshot)``. Each snapshot is an independent CPU tensor with shape
    ``[B, C, H, W]`` so a UI can render it without mutating the sampler state.
    The first callback contains the initial Gaussian noise at step ``0`` and
    the last contains the final sample at ``total_steps``.
    """

    if len(shape) != 4:
        raise ValueError(
            "shape must be [B, C, H, W]"
        )

    batch_size = shape[0]

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape [B, L]"
        )

    if padding_mask.shape != token_ids.shape:
        raise ValueError(
            "padding_mask must match token_ids"
        )

    if padding_mask.dtype != torch.bool:
        raise TypeError(
            "padding_mask must be bool"
        )

    if token_ids.shape[0] != batch_size:
        raise ValueError(
            "text batch size must match shape"
        )

    if token_ids.shape[1] < 2:
        raise ValueError(
            "sequence must contain BOS/EOS"
        )

    if guidance_scale < 0.0:
        raise ValueError(
            "guidance_scale must be non-negative"
        )

    if trajectory_interval <= 0:
        raise ValueError(
            "trajectory_interval must be positive"
        )

    if progress_interval <= 0:
        raise ValueError(
            "progress_interval must be positive"
        )

    token_ids = token_ids.to(
        device=device,
        dtype=torch.long,
    )

    padding_mask = padding_mask.to(
        device=device,
        dtype=torch.bool,
    )

    # ---------------------------------
    # Empty prompt:
    # BOS EOS PAD PAD ...
    # ---------------------------------

    empty_token_ids = torch.full_like(
        token_ids,
        fill_value=pad_id,
    )

    empty_token_ids[:, 0] = bos_id
    empty_token_ids[:, 1] = eos_id

    empty_padding_mask = (
        empty_token_ids != pad_id
    )

    model.eval()

    # x_T ~ N(0, I)
    x_t = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    trajectory = []

    if return_trajectory:
        trajectory.append(
            x_t.detach()
            .cpu()
            .clone()
        )

    total_steps = scheduler.num_timesteps
    if progress_callback is not None:
        progress_callback(
            0,
            total_steps,
            total_steps,
            x_t.detach().cpu().clone(),
        )

    for timestep in reversed(
        range(
            scheduler.num_timesteps
        )
    ):
        timesteps = torch.full(
            (batch_size,),
            fill_value=timestep,
            device=device,
            dtype=torch.long,
        )

        # -----------------------------
        # Conditional prompt
        # -----------------------------

        epsilon_cond = model(
            x_t,
            timesteps,
            token_ids,
            padding_mask,
        )

        # -----------------------------
        # Empty/unconditional prompt
        # -----------------------------

        epsilon_uncond = model(
            x_t,
            timesteps,
            empty_token_ids,
            empty_padding_mask,
        )

        # -----------------------------
        # CFG
        #
        # eps =
        # eps_uncond
        # + s * (eps_cond - eps_uncond)
        # -----------------------------

        epsilon_cfg = (
            classifier_free_guidance(
                epsilon_uncond=epsilon_uncond,
                epsilon_cond=epsilon_cond,
                guidance_scale=guidance_scale,
            )
        )

        (
            model_mean,
            posterior_variance,
            _predicted_x_0,
        ) = scheduler.p_mean_variance(
            model_output=epsilon_cfg,
            x_t=x_t,
            timesteps=timesteps,
            clip_denoised=True,
        )

        noise = torch.randn(
            x_t.shape,
            device=device,
            dtype=x_t.dtype,
            generator=generator,
        )

        nonzero_mask = (
            timesteps > 0
        ).to(
            dtype=x_t.dtype
        ).reshape(
            batch_size,
            1,
            1,
            1,
        )

        x_t = (
            model_mean
            + nonzero_mask
            * torch.sqrt(
                posterior_variance
            )
            * noise
        )

        if not torch.isfinite(
            x_t
        ).all():
            raise FloatingPointError(
                "non-finite sample at "
                f"timestep={timestep}"
            )

        if (
            return_trajectory
            and (
                timestep
                % trajectory_interval
                == 0
                or timestep == 0
            )
        ):
            trajectory.append(
                x_t.detach()
                .cpu()
                .clone()
            )

        completed_steps = total_steps - timestep
        if (
            progress_callback is not None
            and (
                completed_steps % progress_interval == 0
                or timestep == 0
            )
        ):
            progress_callback(
                completed_steps,
                total_steps,
                timestep,
                x_t.detach().cpu().clone(),
            )

    if return_trajectory:
        return (
            x_t,
            trajectory,
        )

    return x_t
