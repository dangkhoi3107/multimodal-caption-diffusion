from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.data.text_product_dataset import (
    TextProductDataset,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.text_conditional_unet import (
    TextConditionalUNet,
)
from src.diffusion.text_trainer import (
    text_training_step,
)
from src.text.vocabulary import (
    Vocabulary,
)


CONFIG_PATH = Path(
    "configs/phase3_text_conditional.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/phase3_overfit_minibatch"
)

STEPS = 1500
LOG_INTERVAL = 50
IMAGES_PER_CLASS = 2


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def select_balanced_indices(
    dataset: TextProductDataset,
) -> list[int]:
    selected = {
        0: [],
        1: [],
        2: [],
    }

    for index, record in enumerate(
        dataset.records
    ):
        class_id = int(
            record["class_id"]
        )

        if (
            class_id in selected
            and len(
                selected[
                    class_id
                ]
            )
            < IMAGES_PER_CLASS
        ):
            selected[
                class_id
            ].append(
                index
            )

        if all(
            len(indices)
            == IMAGES_PER_CLASS
            for indices
            in selected.values()
        ):
            break

    for class_id, indices in (
        selected.items()
    ):
        if len(
            indices
        ) != IMAGES_PER_CLASS:
            raise RuntimeError(
                f"Not enough samples "
                f"for class {class_id}"
            )

    return [
        index
        for class_id
        in sorted(
            selected
        )
        for index
        in selected[
            class_id
        ]
    ]


def main():
    torch.manual_seed(
        42
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if (
        device.type
        == "cuda"
    ):
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    config = load_config()

    vocabulary = Vocabulary.load(
        Path(
            config[
                "text"
            ][
                "vocabulary_path"
            ]
        )
    )

    dataset = TextProductDataset(
        metadata_path=Path(
            config[
                "data"
            ][
                "train_metadata"
            ]
        ),
        vocabulary=vocabulary,
        max_length=int(
            config[
                "text"
            ][
                "max_length"
            ]
        ),
    )

    indices = (
        select_balanced_indices(
            dataset
        )
    )

    items = [
        dataset[
            index
        ]
        for index in indices
    ]

    x_0 = torch.stack(
        [
            item[
                "image"
            ]
            for item
            in items
        ]
    ).to(
        device
    )

    token_ids = torch.stack(
        [
            item[
                "token_ids"
            ]
            for item
            in items
        ]
    ).to(
        device
    )

    padding_mask = torch.stack(
        [
            item[
                "padding_mask"
            ]
            for item
            in items
        ]
    ).to(
        device
    )

    class_ids = [
        int(
            item[
                "class_id"
            ].item()
        )
        for item
        in items
    ]

    captions = [
        item[
            "caption"
        ]
        for item
        in items
    ]

    print(
        "Fixed batch:",
        tuple(
            x_0.shape
        ),
    )

    print(
        "Class IDs:",
        class_ids,
    )

    print(
        "Captions:"
    )

    for class_id, caption in zip(
        class_ids,
        captions,
    ):
        print(
            f"  class={class_id}: "
            f"{caption}"
        )

    model = TextConditionalUNet(
        vocab_size=len(
            vocabulary
        ),
        pad_id=(
            vocabulary.pad_id
        ),
        max_length=int(
            config[
                "text"
            ][
                "max_length"
            ]
        ),
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

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config[
                "training"
            ][
                "learning_rate"
            ]
        ),
    )

    generator = torch.Generator(
        device=device.type
    )

    generator.manual_seed(
        1234
    )

    losses = []
    dropout_fractions = []

    model.train()

    for step in range(
        1,
        STEPS + 1,
    ):
        metrics = text_training_step(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            token_ids=token_ids,
            padding_mask=padding_mask,
            optimizer=optimizer,
            prompt_dropout=float(
                config[
                    "conditioning"
                ][
                    "prompt_dropout"
                ]
            ),
            bos_id=(
                vocabulary.bos_id
            ),
            eos_id=(
                vocabulary.eos_id
            ),
            pad_id=(
                vocabulary.pad_id
            ),
            generator=generator,
        )

        losses.append(
            metrics[
                "loss"
            ]
        )

        dropout_fractions.append(
            metrics[
                "dropped_fraction"
            ]
        )

        if (
            step == 1
            or step
            % LOG_INTERVAL
            == 0
        ):
            print(
                f"step={step:04d} "
                f"loss="
                f"{metrics['loss']:.6f} "
                f"dropout="
                f"{metrics['dropped_fraction']:.3f}"
            )

    window = min(
        100,
        len(
            losses
        ),
    )

    initial_mean = sum(
        losses[
            :window
        ]
    ) / window

    final_mean = sum(
        losses[
            -window:
        ]
    ) / window

    ratio = (
        final_mean
        / initial_mean
    )

    mean_dropout = sum(
        dropout_fractions
    ) / len(
        dropout_fractions
    )

    print()
    print(
        "Initial mean loss:",
        initial_mean,
    )

    print(
        "Final mean loss:",
        final_mean,
    )

    print(
        "Final / initial:",
        ratio,
    )

    print(
        "Mean dropout fraction:",
        mean_dropout,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "config": config,
            "vocabulary": (
                vocabulary.to_dict()
            ),
            "captions": captions,
            "class_ids": (
                class_ids
            ),
            "step": STEPS,
        },
        OUTPUT_ROOT
        / "model.pt",
    )

    report = {
        "steps": STEPS,
        "batch_size": len(
            items
        ),
        "class_ids": (
            class_ids
        ),
        "captions": (
            captions
        ),
        "parameter_count": (
            parameter_count
        ),
        "initial_mean_loss": (
            initial_mean
        ),
        "final_mean_loss": (
            final_mean
        ),
        "final_over_initial": (
            ratio
        ),
        "mean_prompt_dropout": (
            mean_dropout
        ),
    }

    with (
        OUTPUT_ROOT
        / "report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Saved:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()