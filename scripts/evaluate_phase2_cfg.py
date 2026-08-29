from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.diffusion.conditional_sampler import (
    sample_ddpm_cfg,
)
from src.diffusion.conditional_unet import (
    ConditionalUNet,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


CHECKPOINT_PATH = Path(
    "outputs/phase2_overfit_minibatch/model.pt"
)

OUTPUT_ROOT = Path(
    "outputs/phase2_cfg_evaluation"
)

GUIDANCE_SCALES = [
    0.0,
    1.0,
    3.0,
    5.0,
]

CLASS_IDS = [
    0,
    1,
    2,
]

CLASS_NAMES = {
    0: "Dove body serum",
    1: "Dove deodorant",
    2: "Lifebuoy handwash",
}

SEED = 342


def tensor_to_image(
    tensor: torch.Tensor,
):
    image = (
        tensor
        .detach()
        .cpu()
        .clamp(
            -1.0,
            1.0,
        )
    )

    image = (
        image + 1.0
    ) / 2.0

    image = (
        image
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )

    return image


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

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
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

    print(
        "Number of classes:",
        num_classes,
    )

    print(
        "Null class ID:",
        checkpoint[
            "null_class_id"
        ],
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = ConditionalUNet(
        num_classes=num_classes,
        in_channels=int(
            config[
                "model"
            ][
                "in_channels"
            ]
        ),
        out_channels=int(
            config[
                "model"
            ][
                "out_channels"
            ]
        ),
        base_channels=int(
            config[
                "model"
            ][
                "base_channels"
            ]
        ),
        time_embedding_dim=int(
            config[
                "model"
            ][
                "time_embedding_dim"
            ]
        ),
        time_dim=int(
            config[
                "model"
            ][
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

    # -----------------------------
    # Scheduler
    # -----------------------------

    scheduler = DDPMScheduler(
        num_timesteps=int(
            config[
                "diffusion"
            ][
                "num_timesteps"
            ]
        ),
        beta_start=float(
            config[
                "diffusion"
            ][
                "beta_start"
            ]
        ),
        beta_end=float(
            config[
                "diffusion"
            ][
                "beta_end"
            ]
        ),
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    # -----------------------------
    # Sampling
    # -----------------------------

    for guidance_scale in (
        GUIDANCE_SCALES
    ):
        print()
        print(
            "=" * 70
        )

        print(
            "Guidance scale:",
            guidance_scale,
        )

        results[
            guidance_scale
        ] = {}

        for class_id in CLASS_IDS:
            print(
                f"Sampling class "
                f"{class_id}..."
            )

            # CRITICAL:
            # reset exact same RNG sequence
            # for every class.
            generator = torch.Generator(
                device=device.type
            )

            generator.manual_seed(
                SEED
            )

            class_tensor = torch.tensor(
                [class_id],
                dtype=torch.long,
                device=device,
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
                    guidance_scale
                ),
                generator=generator,
            )

            sample = (
                sample[0]
                .detach()
                .cpu()
            )

            results[
                guidance_scale
            ][
                class_id
            ] = sample

            print(
                f"  min="
                f"{sample.min().item():.4f} "
                f"max="
                f"{sample.max().item():.4f} "
                f"mean="
                f"{sample.mean().item():.4f} "
                f"std="
                f"{sample.std().item():.4f}"
            )

    # -----------------------------
    # Numeric class differences
    # -----------------------------

    print()
    print("=" * 70)
    print("PAIRWISE CLASS DIFFERENCES")
    print("=" * 70)

    for guidance_scale in (
        GUIDANCE_SCALES
    ):
        sample_0 = results[
            guidance_scale
        ][0]

        sample_1 = results[
            guidance_scale
        ][1]

        sample_2 = results[
            guidance_scale
        ][2]

        diff_01 = (
            sample_0
            - sample_1
        ).abs().mean().item()

        diff_02 = (
            sample_0
            - sample_2
        ).abs().mean().item()

        diff_12 = (
            sample_1
            - sample_2
        ).abs().mean().item()

        print(
            f"scale={guidance_scale:.1f} "
            f"| 0-1={diff_01:.6f} "
            f"| 0-2={diff_02:.6f} "
            f"| 1-2={diff_12:.6f}"
        )

    # -----------------------------
    # Grid
    # rows = guidance scale
    # columns = class
    # -----------------------------

    figure, axes = plt.subplots(
        nrows=len(
            GUIDANCE_SCALES
        ),
        ncols=len(
            CLASS_IDS
        ),
        figsize=(
            10,
            13,
        ),
    )

    for row, guidance_scale in enumerate(
        GUIDANCE_SCALES
    ):
        for column, class_id in enumerate(
            CLASS_IDS
        ):
            axis = axes[
                row,
                column,
            ]

            sample = results[
                guidance_scale
            ][
                class_id
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
                    CLASS_NAMES[
                        class_id
                    ],
                    fontsize=10,
                )

            if column == 0:
                axis.set_ylabel(
                    f"CFG={guidance_scale}",
                    fontsize=10,
                )

    figure.suptitle(
        (
            "Phase 2 CFG sanity check\n"
            f"Fixed stochastic path, "
            f"seed={SEED}"
        ),
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / "cfg_class_grid.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        "Saved grid:",
        output_path,
    )


if __name__ == "__main__":
    main()