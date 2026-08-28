import torch
from torch import nn
from torch.nn import functional as F

from src.diffusion.scheduler import (
    DDPMScheduler,
)


def compute_diffusion_loss(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    if x_0.ndim != 4:
        raise ValueError(
            "x_0 must have shape [B, C, H, W]"
        )

    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps must have shape [B]"
        )

    if timesteps.shape[0] != x_0.shape[0]:
        raise ValueError(
            "timesteps batch size must match x_0"
        )

    if noise.shape != x_0.shape:
        raise ValueError(
            "noise must have the same shape as x_0"
        )

    if timesteps.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "timesteps must have integer dtype"
        )

    timesteps = timesteps.to(
        device=x_0.device
    )

    noise = noise.to(
        device=x_0.device,
        dtype=x_0.dtype,
    )

    # thêm noise
    x_t = scheduler.q_sample(
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    predicted_noise = model(
        x_t,
        timesteps,
    )

    if predicted_noise.shape != noise.shape:
        raise ValueError(
            "model output must have "
            "the same shape as noise"
        )

    # MSE
    loss = F.mse_loss( 
        predicted_noise,
        noise,
    )

    return loss



def training_step(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    if x_0.ndim != 4:
        raise ValueError(
            "x_0 must have shape [B, C, H, W]"
        )

    batch_size = x_0.shape[0]

    if batch_size <= 0:
        raise ValueError(
            "batch must contain at least one sample"
        )

    # Mỗi ảnh trong batch có timestep riêng.
    timesteps = torch.randint(
        low=0,
        high=scheduler.num_timesteps,
        size=(batch_size,),
        device=x_0.device,
        dtype=torch.long,
        generator=generator,
    )

    # Gaussian noise epsilon ~ N(0, I)
    noise = torch.randn(
        x_0.shape,
        device=x_0.device,
        dtype=x_0.dtype,
        generator=generator,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss = compute_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "diffusion loss is NaN or Inf"
        )

    loss.backward() # cập nhật trọng số qua skip?

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():
            raise FloatingPointError(
                f"non-finite gradient in {name}"
            )

    optimizer.step()

    return {
        "loss": float(
            loss.detach().item()
        ),
        "batch_size": batch_size,
        "timestep_min": int(
            timesteps.min().item()
        ),
        "timestep_max": int(
            timesteps.max().item()
        ),
    }

def train_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x_0 = batch["image"].to(
            device=device,
            dtype=torch.float32,
        )

        metrics = training_step(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            optimizer=optimizer,
            generator=generator,
        )

        batch_size = x_0.shape[0]

        total_loss += (
            metrics["loss"]
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "training dataloader is empty"
        )

    mean_loss = (
        total_loss
        / total_samples
    )

    return {
        "loss": mean_loss,
        "num_samples": total_samples,
    }

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x_0 = batch["image"].to(
            device=device,
            dtype=torch.float32,
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

        loss = compute_diffusion_loss(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "validation loss is NaN or Inf"
            )

        total_loss += (
            float(loss.item())
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
        "num_samples": total_samples,
    }