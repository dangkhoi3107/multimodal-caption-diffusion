import torch

from src.diffusion.scheduler import (
    DDPMScheduler,
)


def classifier_free_guidance(
    epsilon_uncond: torch.Tensor,
    epsilon_cond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    if (
        epsilon_uncond.shape
        != epsilon_cond.shape
    ):
        raise ValueError(
            "conditional and unconditional "
            "predictions must have same shape"
        )

    if guidance_scale < 0.0:
        raise ValueError(
            "guidance_scale must be non-negative"
        )

    return (
        epsilon_uncond
        + guidance_scale
        * (
            epsilon_cond
            - epsilon_uncond
        )
    )


@torch.no_grad()
def sample_ddpm_cfg(
    model,
    scheduler: DDPMScheduler,
    class_ids: torch.Tensor,
    shape: tuple[int, int, int, int],
    device: torch.device,
    guidance_scale: float = 3.0,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    trajectory_interval: int = 100,
):
    if len(shape) != 4:
        raise ValueError(
            "shape must be [B, C, H, W]"
        )

    batch_size = shape[0]

    if class_ids.shape != (
        batch_size,
    ):
        raise ValueError(
            "class_ids must have shape [B]"
        )

    if guidance_scale < 0.0:
        raise ValueError(
            "guidance_scale must be non-negative"
        )

    if trajectory_interval <= 0:
        raise ValueError(
            "trajectory_interval must be positive"
        )

    class_ids = class_ids.to(
        device=device,
        dtype=torch.long,
    )

    null_class_ids = torch.full(
        (batch_size,),
        fill_value=model.null_class_id,
        device=device,
        dtype=torch.long,
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
            x_t.detach().cpu().clone()
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

        # ---------------------------------
        # Conditional prediction
        # ---------------------------------

        epsilon_cond = model(
            x_t,
            timesteps,
            class_ids,
        )

        # ---------------------------------
        # Unconditional prediction
        # ---------------------------------

        epsilon_uncond = model(
            x_t,
            timesteps,
            null_class_ids,
        )

        # ---------------------------------
        # CFG
        # ---------------------------------

        epsilon_cfg = classifier_free_guidance(
            epsilon_uncond=epsilon_uncond,
            epsilon_cond=epsilon_cond,
            guidance_scale=guidance_scale,
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
                f"non-finite sample at "
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

    if return_trajectory:
        return (
            x_t,
            trajectory,
        )

    return x_t