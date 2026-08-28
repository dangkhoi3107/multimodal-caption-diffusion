import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.data.product_dataset import ProductImageDataset
from src.diffusion.sampler import sample_ddpm
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.unet import UNet


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

CHECKPOINT_PATH = Path(
    "outputs/phase1_overfit_one_image/model.pt"
)

OUTPUT_DIR = Path(
    "outputs/phase1_overfit_one_image"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def to_image(
    tensor: torch.Tensor,
):
    tensor = tensor.detach().cpu()
    tensor = tensor.clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0

    return tensor.permute(
        1,
        2,
        0,
    ).numpy()


def main() -> None:
    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # -------------------------
    # Original training image
    # -------------------------

    dataset = ProductImageDataset(
        Path(
            config["data"][
                "train_metadata"
            ]
        )
    )

    target = dataset[0]["image"]

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
            config["model"]["in_channels"]
        ),
        out_channels=int(
            config["model"]["out_channels"]
        ),
        base_channels=int(
            config["model"]["base_channels"]
        ),
        time_embedding_dim=int(
            config["model"][
                "time_embedding_dim"
            ]
        ),
        time_dim=int(
            config["model"]["time_dim"]
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

    # -------------------------
    # Fixed-noise generation
    # -------------------------

    sample_seed = 4242

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        sample_seed
    )

    sample, trajectory = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(1, 3, 64, 64),
        device=device,
        generator=generator,
        return_trajectory=True,
        trajectory_interval=100,
    )

    if not torch.isfinite(sample).all():
        raise RuntimeError(
            "Generated sample contains NaN/Inf"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Target vs generated
    # -------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(5, 2.8),
    )

    axes[0].imshow(
        to_image(target)
    )
    axes[0].set_title(
        "Training image"
    )
    axes[0].axis("off")

    axes[1].imshow(
        to_image(sample[0])
    )
    axes[1].set_title(
        f"Fixed noise\nseed={sample_seed}"
    )
    axes[1].axis("off")

    plt.tight_layout()

    comparison_path = (
        OUTPUT_DIR
        / "fixed_noise_sample.png"
    )

    plt.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------
    # Denoising trajectory
    # -------------------------

    num_states = len(
        trajectory
    )

    fig, axes = plt.subplots(
        1,
        num_states,
        figsize=(
            num_states * 1.6,
            2.2,
        ),
    )

    if num_states == 1:
        axes = [axes]

    for index, state in enumerate(
        trajectory
    ):
        axes[index].imshow(
            to_image(
                state[0]
            )
        )

        if index == 0:
            title = "noise"
        elif index == num_states - 1:
            title = "final"
        else:
            title = f"state {index}"

        axes[index].set_title(
            title,
            fontsize=8,
        )

        axes[index].axis(
            "off"
        )

    plt.tight_layout()

    trajectory_path = (
        OUTPUT_DIR
        / "fixed_noise_trajectory.png"
    )

    plt.savefig(
        trajectory_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------
    # Metadata
    # -------------------------

    metadata = {
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "sample_seed": sample_seed,
        "sample_shape": list(
            sample.shape
        ),
        "sample_min": float(
            sample.min().item()
        ),
        "sample_max": float(
            sample.max().item()
        ),
        "sample_mean": float(
            sample.mean().item()
        ),
        "sample_std": float(
            sample.std().item()
        ),
        "finite": bool(
            torch.isfinite(
                sample
            ).all()
        ),
    }

    metadata_path = (
        OUTPUT_DIR
        / "fixed_noise_sample.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        "Saved:",
        comparison_path,
    )

    print(
        "Saved:",
        trajectory_path,
    )

    print(
        "Saved:",
        metadata_path,
    )

    print()

    for key, value in metadata.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()