from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.product_dataset import ProductImageDataset
from src.diffusion.conditional_sampler import sample_ddpm_cfg
from src.diffusion.conditional_unet import ConditionalUNet
from src.diffusion.scheduler import DDPMScheduler


CHECKPOINT_PATH = Path(
    "outputs/phase2_conditional/best.pt"
)

TRAIN_METADATA = Path(
    "data/processed/products_multiclass_64/train.jsonl"
)

OUTPUT_ROOT = Path(
    "outputs/phase2_quality_evaluation"
)

CLASS_NAMES = {
    0: "Dove body serum",
    1: "Dove deodorant",
    2: "Lifebuoy handwash",
}

SEEDS = [
    101,
    202,
    303,
    404,
]

GUIDANCE_SCALE = 2.0


def tensor_to_image(
    tensor: torch.Tensor,
):
    image = (
        tensor
        .detach()
        .cpu()
        .clamp(-1.0, 1.0)
    )

    image = (
        image + 1.0
    ) / 2.0

    return (
        image
        .permute(1, 2, 0)
        .numpy()
    )


def pairwise_diversity(
    samples: list[torch.Tensor],
) -> float:
    distances = []

    for i in range(
        len(samples)
    ):
        for j in range(
            i + 1,
            len(samples),
        ):
            distance = (
                samples[i]
                - samples[j]
            ).abs().mean().item()

            distances.append(
                distance
            )

    return sum(
        distances
    ) / len(
        distances
    )


def nearest_train_distance(
    sample: torch.Tensor,
    train_images: torch.Tensor,
) -> tuple[float, int]:
    sample = sample.unsqueeze(0)

    distances = (
        train_images
        - sample
    ).abs().mean(
        dim=(1, 2, 3)
    )

    value, index = torch.min(
        distances,
        dim=0,
    )

    return (
        float(value.item()),
        int(index.item()),
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint[
        "config"
    ]

    num_classes = int(
        checkpoint[
            "num_classes"
        ]
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = ConditionalUNet(
        num_classes=num_classes,
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
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

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

    # ---------------------------------
    # Train images for NN sanity check
    # ---------------------------------

    train_dataset = ProductImageDataset(
        TRAIN_METADATA
    )

    train_images = torch.stack(
        [
            train_dataset[index][
                "image"
            ]
            for index in range(
                len(train_dataset)
            )
        ],
        dim=0,
    )

    print(
        "Train images:",
        tuple(
            train_images.shape
        ),
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    # ---------------------------------
    # Same class, different seeds
    # ---------------------------------

    for class_id in range(
        num_classes
    ):
        print()
        print(
            "=" * 70
        )

        print(
            f"Class {class_id}: "
            f"{CLASS_NAMES[class_id]}"
        )

        class_samples = []

        for seed in SEEDS:
            generator = torch.Generator(
                device=device.type
            )

            generator.manual_seed(
                seed
            )

            class_tensor = torch.tensor(
                [class_id],
                device=device,
                dtype=torch.long,
            )

            sample = sample_ddpm_cfg(
                model=model,
                scheduler=scheduler,
                class_ids=class_tensor,
                shape=(
                    1,
                    3,
                    64,
                    64,
                ),
                device=device,
                guidance_scale=(
                    GUIDANCE_SCALE
                ),
                generator=generator,
            )[0].cpu()

            class_samples.append(
                sample
            )

            (
                nearest_distance,
                nearest_index,
            ) = nearest_train_distance(
                sample=sample,
                train_images=(
                    train_images
                ),
            )

            nearest_class = int(
                train_dataset[
                    nearest_index
                ][
                    "class_id"
                ].item()
            )

            nearest_file = (
                train_dataset[
                    nearest_index
                ][
                    "file_name"
                ]
            )

            print(
                f"seed={seed} "
                f"nearest_L1="
                f"{nearest_distance:.6f} "
                f"nearest_class="
                f"{nearest_class} "
                f"nearest_file="
                f"{nearest_file}"
            )

        diversity = (
            pairwise_diversity(
                class_samples
            )
        )

        print(
            "Mean pairwise diversity:",
            f"{diversity:.6f}",
        )

        results[
            class_id
        ] = class_samples

    # ---------------------------------
    # Same-class different-seed grid
    # ---------------------------------

    figure, axes = plt.subplots(
        nrows=num_classes,
        ncols=len(SEEDS),
        figsize=(
            12,
            9,
        ),
    )

    for row in range(
        num_classes
    ):
        for column, seed in enumerate(
            SEEDS
        ):
            axis = axes[
                row,
                column,
            ]

            sample = results[
                row
            ][
                column
            ]

            axis.imshow(
                tensor_to_image(
                    sample
                )
            )

            axis.axis(
                "off"
            )

            if row == 0:
                axis.set_title(
                    f"seed={seed}"
                )

            if column == 0:
                axis.set_ylabel(
                    CLASS_NAMES[row]
                )

    figure.suptitle(
        (
            "Phase 2 — Same class, "
            "different seeds\n"
            f"CFG={GUIDANCE_SCALE}"
        )
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / "same_class_different_seed.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()