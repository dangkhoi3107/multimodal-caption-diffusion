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
    "outputs/phase3_overfit_minibatch/model.pt"
)

OUTPUT_ROOT = Path(
    "outputs/phase3_cfg_evaluation"
)

GUIDANCE_SCALES = [
    0.0,
    1.0,
    2.0,
    3.0,
]

PROMPTS = {
    0: "a dove body serum in a white bottle",
    1: "a dove deodorant in a blue tube",
    2: "a red lifebuoy handwash pouch",
}

PROMPT_NAMES = {
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

    config = checkpoint["config"]

    vocabulary = restore_vocabulary(
        checkpoint["vocabulary"]
    )

    max_length = int(
        config["text"]["max_length"]
    )

    print(
        "Vocabulary size:",
        len(vocabulary),
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = TextConditionalUNet(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        max_length=max_length,
        text_embedding_dim=int(
            config["text"]["embedding_dim"]
        ),
        text_num_heads=int(
            config["text"]["num_heads"]
        ),
        text_num_layers=int(
            config["text"]["num_layers"]
        ),
        text_feedforward_dim=int(
            config["text"]["feedforward_dim"]
        ),
        text_dropout=float(
            config["text"]["dropout"]
        ),
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
            config["model"]["time_embedding_dim"]
        ),
        time_dim=int(
            config["model"]["time_dim"]
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # ---------------------------------
    # Scheduler
    # ---------------------------------

    scheduler = DDPMScheduler(
        num_timesteps=int(
            config["diffusion"]["num_timesteps"]
        ),
        beta_start=float(
            config["diffusion"]["beta_start"]
        ),
        beta_end=float(
            config["diffusion"]["beta_end"]
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
        print("=" * 70)

        print(
            "Guidance scale:",
            guidance_scale,
        )

        results[
            guidance_scale
        ] = {}

        for prompt_id, prompt in PROMPTS.items():
            print(
                f"Prompt {prompt_id}: "
                f"{prompt}"
            )

            ids = encode(
                text=prompt,
                vocabulary=vocabulary,
                max_length=max_length,
            ).unsqueeze(0)

            mask = padding_mask(
                ids,
                vocabulary,
            )

            # Same initial noise AND
            # same reverse stochastic path
            # for every prompt.
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
                bos_id=vocabulary.bos_id,
                eos_id=vocabulary.eos_id,
                pad_id=vocabulary.pad_id,
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
    # Pairwise prompt differences
    # ---------------------------------

    print()
    print("=" * 70)
    print("PAIRWISE PROMPT DIFFERENCES")
    print("=" * 70)

    pairwise_report = {}

    for guidance_scale in GUIDANCE_SCALES:
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
            sample_0 - sample_1
        ).abs().mean().item()

        diff_02 = (
            sample_0 - sample_2
        ).abs().mean().item()

        diff_12 = (
            sample_1 - sample_2
        ).abs().mean().item()

        print(
            f"scale={guidance_scale:.1f} "
            f"| 0-1={diff_01:.6f} "
            f"| 0-2={diff_02:.6f} "
            f"| 1-2={diff_12:.6f}"
        )

        pairwise_report[
            str(guidance_scale)
        ] = {
            "0-1": diff_01,
            "0-2": diff_02,
            "1-2": diff_12,
        }

    # ---------------------------------
    # Image grid
    # ---------------------------------

    figure, axes = plt.subplots(
        nrows=len(
            GUIDANCE_SCALES
        ),
        ncols=len(
            PROMPTS
        ),
        figsize=(10, 13),
    )

    for row, guidance_scale in enumerate(
        GUIDANCE_SCALES
    ):
        for column, prompt_id in enumerate(
            PROMPTS
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

            if row == 0:
                axis.set_title(
                    PROMPT_NAMES[
                        prompt_id
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
            "Phase 3 Text CFG sanity check\n"
            f"Fixed stochastic path, seed={SEED}"
        ),
        fontsize=14,
    )

    plt.tight_layout()

    grid_path = (
        OUTPUT_ROOT
        / "cfg_prompt_grid.png"
    )

    plt.savefig(
        grid_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    report = {
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "seed": SEED,
        "guidance_scales": (
            GUIDANCE_SCALES
        ),
        "prompts": PROMPTS,
        "pairwise_mean_absolute_difference": (
            pairwise_report
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