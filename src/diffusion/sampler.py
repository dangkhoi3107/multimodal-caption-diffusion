import torch
from torch import nn

from src.diffusion.scheduler import DDPMScheduler


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, int, int, int],
    device: torch.device,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    trajectory_interval: int = 100,
):
    """
    Run the full DDPM reverse process starting from Gaussian noise.

    Expected model signature:
        model(x_t, timesteps) -> predicted_noise

    Args:
        model:
            Trained noise-prediction model.
        scheduler:
            DDPMScheduler with p_sample().
        shape:
            Output shape [B, C, H, W].
        device:
            Device used for sampling.
        generator:
            Optional torch.Generator for deterministic sampling.
        return_trajectory:
            If True, return selected intermediate states.
        trajectory_interval:
            Save one trajectory state every N timesteps.

    Returns:
        final_sample, or (final_sample, trajectory) when
        return_trajectory=True.
    """
    if len(shape) != 4:
        raise ValueError(
            "shape must be [B, C, H, W]"
        )

    if any(size <= 0 for size in shape):
        raise ValueError(
            "all shape dimensions must be positive"
        )

    if trajectory_interval <= 0:
        raise ValueError(
            "trajectory_interval must be positive"
        )

    batch_size = shape[0]

    model.eval()

    # Start from x_T ~ N(0, I).
    x_t = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    trajectory: list[torch.Tensor] = []

    if return_trajectory:
        trajectory.append(
            x_t.detach().cpu().clone()
        )

    # Reverse process: T-1 -> ... -> 1 -> 0.
    for timestep in reversed(
        range(scheduler.num_timesteps)
    ):
        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        x_t = scheduler.p_sample(
            model=model,
            x_t=x_t,
            timesteps=timesteps,
            generator=generator,
        )

        if not torch.isfinite(x_t).all():
            raise FloatingPointError(
                f"Non-finite sample at timestep {timestep}"
            )

        if return_trajectory:
            if (
                timestep % trajectory_interval == 0
                or timestep == 0
            ):
                trajectory.append(
                    x_t.detach().cpu().clone()
                )

    if return_trajectory:
        return x_t, trajectory

    return x_t
