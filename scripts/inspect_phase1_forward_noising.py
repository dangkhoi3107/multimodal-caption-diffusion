import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.data.product_dataset import ProductImageDataset
from src.diffusion.scheduler import DDPMScheduler


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

OUTPUT_DIR = Path(
    "outputs/phase1_unconditional"
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
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0

    return tensor.permute(
        1,
        2,
        0,
    ).numpy()


def main() -> None:
    config = load_config()

    seed = int(
        config["training"]["seed"]
    )

    torch.manual_seed(seed)

    dataset = ProductImageDataset(
        Path(
            config["data"][
                "train_metadata"
            ]
        )
    )

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------
    # One-image visualization
    # --------------------------------

    x_0 = dataset[0]["image"].unsqueeze(0)

    generator = torch.Generator()
    generator.manual_seed(seed + 500)

    # Dùng cùng một noise cho mọi timestep
    # để so sánh forward process công bằng.
    noise = torch.randn(
        x_0.shape,
        generator=generator,
        dtype=x_0.dtype,
    )

    timesteps_to_show = [
        0,
        100,
        300,
        500,
        999,
    ]

    states = []

    for timestep in timesteps_to_show:
        t = torch.tensor(
            [timestep],
            dtype=torch.long,
        )

        x_t = scheduler.q_sample(
            x_0=x_0,
            timesteps=t,
            noise=noise,
        )

        states.append(
            x_t[0]
        )

    fig, axes = plt.subplots(
        1,
        len(states) + 1,
        figsize=(12, 2.5),
    )

    axes[0].imshow(
        to_image(x_0[0])
    )
    axes[0].set_title("clean x0")
    axes[0].axis("off")

    for axis, state, timestep in zip(
        axes[1:],
        states,
        timesteps_to_show,
    ):
        axis.imshow(
            to_image(state)
        )
        axis.set_title(
            f"t={timestep}"
        )
        axis.axis("off")

    plt.tight_layout()

    figure_path = (
        OUTPUT_DIR
        / "forward_noising.png"
    )

    plt.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------
    # t=999 statistics
    # --------------------------------

    num_images = min(
        32,
        len(dataset),
    )

    batch = torch.stack(
        [
            dataset[index]["image"]
            for index in range(num_images)
        ],
        dim=0,
    )

    stats_generator = torch.Generator()
    stats_generator.manual_seed(
        seed + 501
    )

    final_noise = torch.randn(
        batch.shape,
        generator=stats_generator,
        dtype=batch.dtype,
    )

    final_timesteps = torch.full(
        (num_images,),
        999,
        dtype=torch.long,
    )

    x_999 = scheduler.q_sample(
        x_0=batch,
        timesteps=final_timesteps,
        noise=final_noise,
    )

    alpha_bar_999 = float(
        scheduler.alpha_bars[999].item()
    )

    stats = {
        "num_images": num_images,
        "timestep": 999,
        "alpha_bar_999": alpha_bar_999,
        "x999_mean": float(
            x_999.mean().item()
        ),
        "x999_std": float(
            x_999.std().item()
        ),
        "noise_mean": float(
            final_noise.mean().item()
        ),
        "noise_std": float(
            final_noise.std().item()
        ),
        "mean_absolute_difference_from_noise": float(
            (
                x_999
                - final_noise
            )
            .abs()
            .mean()
            .item()
        ),
    }

    stats_path = (
        OUTPUT_DIR
        / "forward_noising_stats.json"
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            stats,
            file,
            indent=2,
        )

    print(
        "Saved:",
        figure_path,
    )

    print(
        "Saved:",
        stats_path,
    )

    print()

    for key, value in stats.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()