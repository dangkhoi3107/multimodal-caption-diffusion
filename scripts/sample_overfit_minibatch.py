from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.diffusion.sampler import sample_ddpm
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.unet import UNet


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

CHECKPOINT_PATH = Path(
    "outputs/phase1_overfit_minibatch/model.pt"
)

OUTPUT_DIR = Path(
    "outputs/phase1_overfit_minibatch"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def tensor_to_image(
    tensor: torch.Tensor,
) -> torch.Tensor:
    # Chỉ clamp khi visualize.
    tensor = tensor.clamp(
        -1.0,
        1.0,
    )

    tensor = (
        tensor + 1.0
    ) / 2.0

    return tensor


def save_sample_grid(
    samples: torch.Tensor,
    output_path: Path,
) -> None:
    samples = tensor_to_image(
        samples.detach().cpu()
    )

    num_images = samples.shape[0]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(8, 4),
    )

    axes = axes.reshape(-1)

    for index in range(
        num_images
    ):
        image = samples[index].permute(
            1,
            2,
            0,
        )

        axes[index].imshow(
            image.numpy(),
            interpolation="nearest",
        )

        axes[index].set_title(
            f"sample {index}"
        )

        axes[index].axis("off")

    for axis in axes[num_images:]:
        axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_trajectory(
    trajectory: list[torch.Tensor],
    output_path: Path,
) -> None:
    # Chỉ xem sample đầu tiên.
    states = [
        tensor_to_image(state[0])
        for state in trajectory
    ]

    num_states = len(states)

    fig, axes = plt.subplots(
        1,
        num_states,
        figsize=(
            num_states * 2,
            2.5,
        ),
    )

    if num_states == 1:
        axes = [axes]

    for index, (
        axis,
        state,
    ) in enumerate(
        zip(
            axes,
            states,
        )
    ):
        image = state.permute(
            1,
            2,
            0,
        )

        axis.imshow(
            image.numpy(),
            interpolation="nearest",
        )

        if index == 0:
            title = "noise"
        else:
            title = f"state {index}"

        axis.set_title(
            title,
            fontsize=8,
        )

        axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main() -> None:
    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler = DDPMScheduler(
        num_timesteps=int(
            config["diffusion"][
                "num_timesteps"
            ]
        ),
        beta_start=float(
            config["diffusion"][
                "beta_start"
            ]
        ),
        beta_end=float(
            config["diffusion"][
                "beta_end"
            ]
        ),
    )

    # -------------------------
    # Model
    # -------------------------

    model = UNet(
        in_channels=int(
            config["model"][
                "in_channels"
            ]
        ),
        out_channels=int(
            config["model"][
                "out_channels"
            ]
        ),
        base_channels=int(
            config["model"][
                "base_channels"
            ]
        ),
        time_embedding_dim=int(
            config["model"][
                "time_embedding_dim"
            ]
        ),
        time_dim=int(
            config["model"][
                "time_dim"
            ]
        ),
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Loaded checkpoint:",
        CHECKPOINT_PATH,
    )

    # -------------------------
    # Fixed generator
    # -------------------------

    seed = 42

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )

    print(
        "Sampling seed:",
        seed,
    )

    # -------------------------
    # Reverse DDPM
    # -------------------------

    samples, trajectory = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(
            8,
            3,
            64,
            64,
        ),
        device=device,
        generator=generator,
        return_trajectory=True,
        trajectory_interval=100,
    )

    print(
        "Sample shape:",
        tuple(samples.shape),
    )

    print(
        "Sample min:",
        float(samples.min().item()),
    )

    print(
        "Sample max:",
        float(samples.max().item()),
    )

    print(
        "Finite:",
        bool(
            torch.isfinite(
                samples
            ).all()
        ),
    )

    # -------------------------
    # Save
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_path = (
        OUTPUT_DIR
        / "generated_grid.png"
    )

    trajectory_path = (
        OUTPUT_DIR
        / "denoising_trajectory.png"
    )

    save_sample_grid(
        samples=samples,
        output_path=grid_path,
    )

    save_trajectory(
        trajectory=trajectory,
        output_path=trajectory_path,
    )

    print()
    print(
        "Generated grid:",
        grid_path,
    )

    print(
        "Trajectory:",
        trajectory_path,
    )


if __name__ == "__main__":
    main()