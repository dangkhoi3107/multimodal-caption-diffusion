import torch
from torch import nn
from torch.nn import functional as F

from src.diffusion.conditioning import (
    drop_condition,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


def compute_conditional_diffusion_loss(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    if x_0.ndim != 4:
        raise ValueError(
            "x_0 must have shape [B, C, H, W]"
        )

    batch_size = x_0.shape[0]

    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps must have shape [B]"
        )

    if class_ids.ndim != 1:
        raise ValueError(
            "class_ids must have shape [B]"
        )

    if timesteps.shape[0] != batch_size:
        raise ValueError(
            "timesteps batch size must match x_0"
        )

    if class_ids.shape[0] != batch_size:
        raise ValueError(
            "class_ids batch size must match x_0"
        )

    if noise.shape != x_0.shape:
        raise ValueError(
            "noise must have same shape as x_0"
        )

    if timesteps.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "timesteps must have integer dtype"
        )

    if class_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "class_ids must have integer dtype"
        )

    timesteps = timesteps.to(
        device=x_0.device,
    )

    class_ids = class_ids.to(
        device=x_0.device,
    )

    noise = noise.to(
        device=x_0.device,
        dtype=x_0.dtype,
    )

    x_t = scheduler.q_sample(
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    predicted_noise = model(
        x_t,
        timesteps,
        class_ids,
    )

    if predicted_noise.shape != noise.shape:
        raise ValueError(
            "model output must have "
            "same shape as noise"
        )

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    return loss


def conditional_training_step(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    class_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    condition_dropout: float,
    null_class_id: int,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    if x_0.ndim != 4:
        raise ValueError(
            "x_0 must have shape [B, C, H, W]"
        )

    batch_size = x_0.shape[0]

    if batch_size <= 0:
        raise ValueError(
            "batch must not be empty"
        )

    if class_ids.shape != (
        batch_size,
    ):
        raise ValueError(
            "class_ids must have shape [B]"
        )

    class_ids = class_ids.to(
        device=x_0.device,
        dtype=torch.long,
    )

    # ---------------------------------
    # Classifier-free guidance dropout
    # ---------------------------------

    model_class_ids = drop_condition(
        class_ids=class_ids,
        probability=condition_dropout,
        null_class_id=null_class_id,
        generator=generator,
    )

    # ---------------------------------
    # Random timestep for each sample
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

    loss = compute_conditional_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
        class_ids=model_class_ids,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "conditional diffusion loss "
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
                f"non-finite gradient in {name}"
            )

    optimizer.step()

    dropped_count = int(
        (
            model_class_ids
            == null_class_id
        )
        .sum()
        .item()
    )

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
        "dropped_count": (
            dropped_count
        ),
        "dropped_fraction": (
            dropped_count
            / batch_size
        ),
    }


def train_conditional_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    condition_dropout: float,
    null_class_id: int,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_dropped = 0

    for batch in dataloader:
        x_0 = batch["image"].to(
            device=device,
            dtype=torch.float32,
        )

        class_ids = batch[
            "class_id"
        ].to(
            device=device,
            dtype=torch.long,
        )

        metrics = conditional_training_step(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            class_ids=class_ids,
            optimizer=optimizer,
            condition_dropout=(
                condition_dropout
            ),
            null_class_id=null_class_id,
            generator=generator,
        )

        batch_size = x_0.shape[0]

        total_loss += (
            metrics["loss"]
            * batch_size
        )

        total_samples += batch_size

        total_dropped += int(
            metrics["dropped_count"]
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
        "num_samples": total_samples,
        "dropped_fraction": (
            total_dropped
            / total_samples
        ),
    }


@torch.no_grad()
def validate_conditional_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, float | int]:
    """Validation uses real class IDs.

    No condition dropout is applied here.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x_0 = batch["image"].to(
            device=device,
            dtype=torch.float32,
        )

        class_ids = batch[
            "class_id"
        ].to(
            device=device,
            dtype=torch.long,
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

        loss = compute_conditional_diffusion_loss(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
            class_ids=class_ids,
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
        "num_samples": (
            total_samples
        ),
    }