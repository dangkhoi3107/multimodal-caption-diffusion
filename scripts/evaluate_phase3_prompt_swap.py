from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.text_conditional_unet import TextConditionalUNet
from src.diffusion.text_sampler import sample_ddpm_text_cfg
from src.text.tokenizer import encode, padding_mask
from src.text.vocabulary import Vocabulary


CHECKPOINT_PATH = Path(
    "outputs/phase3_text_conditional/best.pt"
)

OUTPUT_ROOT = Path(
    "outputs/phase3_prompt_swap_evaluation"
)

# Keep a single guidance scale so the experiment isolates
# the effect of changing one text attribute.
GUIDANCE_SCALES = [
    2.0,
]

# Each adjacent pair changes ONLY the color word.
PROMPTS = {
    0: "a white dove body serum bottle",
    1: "a red dove body serum bottle",

    2: "a blue dove deodorant tube",
    3: "a white dove deodorant tube",

    4: "a red lifebuoy handwash pouch",
    5: "a blue lifebuoy handwash pouch",
}

PROMPT_NAMES = {
    0: "Serum / white",
    1: "Serum / red",

    2: "Deodorant / blue",
    3: "Deodorant / white",

    4: "Lifebuoy / red",
    5: "Lifebuoy / blue",
}

# name -> (baseline_prompt_id, swapped_prompt_id)
SWAP_PAIRS = {
    "serum_white_vs_red": (
        0,
        1,
    ),
    "deodorant_blue_vs_white": (
        2,
        3,
    ),
    "lifebuoy_red_vs_blue": (
        4,
        5,
    ),
}

SEED = 342


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


def restore_vocabulary(
    data: dict,
) -> Vocabulary:
    vocabulary = Vocabulary(
        token_to_id={
            str(token): int(token_id)
            for token, token_id
            in data["token_to_id"].items()
        },
        id_to_token=tuple(
            str(token)
            for token
            in data["id_to_token"]
        ),
    )

    vocabulary.validate()

    return vocabulary


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

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
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

    vocabulary = restore_vocabulary(
        checkpoint[
            "vocabulary"
        ]
    )

    max_length = int(
        config[
            "text"
        ][
            "max_length"
        ]
    )

    print(
        "Vocabulary size:",
        len(vocabulary),
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = TextConditionalUNet(
        vocab_size=len(
            vocabulary
        ),
        pad_id=(
            vocabulary.pad_id
        ),
        max_length=max_length,
        text_embedding_dim=int(
            config[
                "text"
            ][
                "embedding_dim"
            ]
        ),
        text_num_heads=int(
            config[
                "text"
            ][
                "num_heads"
            ]
        ),
        text_num_layers=int(
            config[
                "text"
            ][
                "num_layers"
            ]
        ),
        text_feedforward_dim=int(
            config[
                "text"
            ][
                "feedforward_dim"
            ]
        ),
        text_dropout=float(
            config[
                "text"
            ][
                "dropout"
            ]
        ),
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

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    # ---------------------------------
    # Scheduler
    # ---------------------------------

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

    # ---------------------------------
    # Sampling
    # ---------------------------------

    for guidance_scale in GUIDANCE_SCALES:
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

        for prompt_id, prompt in (
            PROMPTS.items()
        ):
            print(
                f"Prompt {prompt_id}: "
                f"{prompt}"
            )

            ids = encode(
                text=prompt,
                vocabulary=vocabulary,
                max_length=max_length,
            ).unsqueeze(
                0
            )

            mask = padding_mask(
                ids,
                vocabulary,
            )

            # Reset RNG for EVERY prompt.
            # Every prompt starts from the exact same x_T and
            # uses the same reverse-process Gaussian noise sequence.
            # Any output difference is therefore caused by text.
            generator = torch.Generator(
                device=device.type
            )

            generator.manual_seed(
                SEED
            )

            sample = sample_ddpm_text_cfg(
                model=model,
                scheduler=scheduler,
                token_ids=ids,
                padding_mask=mask,
                bos_id=(
                    vocabulary.bos_id
                ),
                eos_id=(
                    vocabulary.eos_id
                ),
                pad_id=(
                    vocabulary.pad_id
                ),
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
                sample[
                    0
                ]
                .detach()
                .cpu()
            )

            results[
                guidance_scale
            ][
                prompt_id
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

    # ---------------------------------
    # Prompt-swap differences
    # ---------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "PROMPT SWAP DIFFERENCES"
    )
    print(
        "=" * 70
    )

    pairwise_report = {}

    for guidance_scale in (
        GUIDANCE_SCALES
    ):
        scale_report = {}

        for (
            swap_name,
            (
                prompt_a,
                prompt_b,
            ),
        ) in SWAP_PAIRS.items():

            sample_a = results[
                guidance_scale
            ][
                prompt_a
            ]

            sample_b = results[
                guidance_scale
            ][
                prompt_b
            ]

            difference = (
                sample_a
                - sample_b
            ).abs().mean().item()

            scale_report[
                swap_name
            ] = {
                "prompt_a_id": (
                    prompt_a
                ),
                "prompt_b_id": (
                    prompt_b
                ),
                "prompt_a": (
                    PROMPTS[
                        prompt_a
                    ]
                ),
                "prompt_b": (
                    PROMPTS[
                        prompt_b
                    ]
                ),
                "mean_absolute_difference": (
                    difference
                ),
            }

            print(
                f"scale="
                f"{guidance_scale:.1f} "
                f"| {swap_name}="
                f"{difference:.6f}"
            )

        pairwise_report[
            str(
                guidance_scale
            )
        ] = scale_report

    # ---------------------------------
    # Image grid
    #
    # 3 rows x 2 columns:
    # Serum:      white | red
    # Deodorant:  blue  | white
    # Lifebuoy:   red   | blue
    # ---------------------------------

    guidance_scale = (
        GUIDANCE_SCALES[
            0
        ]
    )

    grid_pairs = [
        (
            "Dove body serum",
            0,
            1,
        ),
        (
            "Dove deodorant",
            2,
            3,
        ),
        (
            "Lifebuoy handwash",
            4,
            5,
        ),
    ]

    figure, axes = plt.subplots(
        nrows=len(
            grid_pairs
        ),
        ncols=2,
        figsize=(
            8,
            11,
        ),
    )

    for row, (
        product_name,
        prompt_a,
        prompt_b,
    ) in enumerate(
        grid_pairs
    ):
        for column, prompt_id in enumerate(
            (
                prompt_a,
                prompt_b,
            )
        ):
            axis = axes[
                row,
                column,
            ]

            sample = results[
                guidance_scale
            ][
                prompt_id
            ]

            axis.imshow(
                tensor_to_image(
                    sample
                )
            )

            axis.axis(
                "off"
            )

            axis.set_title(
                PROMPT_NAMES[
                    prompt_id
                ],
                fontsize=11,
            )

            if column == 0:
                axis.set_ylabel(
                    product_name,
                    fontsize=11,
                )

    figure.suptitle(
        (
            "Phase 3 Prompt-Swap Evaluation\n"
            f"CFG={guidance_scale}, "
            f"fixed stochastic path, "
            f"seed={SEED}\n"
            "Each row changes only "
            "the color word"
        ),
        fontsize=14,
    )

    plt.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.94,
        )
    )

    grid_path = (
        OUTPUT_ROOT
        / "prompt_swap_grid.png"
    )

    plt.savefig(
        grid_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # ---------------------------------
    # Report
    # ---------------------------------

    report = {
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": (
            checkpoint.get(
                "epoch"
            )
        ),
        "best_valid_loss": (
            checkpoint.get(
                "best_valid_loss"
            )
        ),
        "seed": SEED,
        "guidance_scales": (
            GUIDANCE_SCALES
        ),
        "vocabulary_size": len(
            vocabulary
        ),
        "max_length": (
            max_length
        ),
        "prompts": {
            str(
                prompt_id
            ): prompt
            for (
                prompt_id,
                prompt,
            ) in PROMPTS.items()
        },
        "swap_pairs": {
            name: [
                pair[
                    0
                ],
                pair[
                    1
                ],
            ]
            for (
                name,
                pair,
            ) in SWAP_PAIRS.items()
        },
        "pairwise_mean_absolute_difference": (
            pairwise_report
        ),
        "experiment_note": (
            "Each prompt pair changes only the "
            "color word while keeping brand, "
            "product type, and package fixed. "
            "Because training attributes are "
            "strongly correlated with SKU, "
            "successful text conditioning does "
            "not automatically imply full "
            "compositional generalization."
        ),
    }

    report_path = (
        OUTPUT_ROOT
        / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved grid:",
        grid_path,
    )

    print(
        "Saved report:",
        report_path,
    )


if __name__ == "__main__":
    main()
