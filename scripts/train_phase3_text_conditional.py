from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

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
    train_text_epoch,
    validate_text_epoch,
)
from src.text.vocabulary import (
    Vocabulary,
)


CONFIG_PATH = Path(
    "configs/phase3_text_conditional.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/phase3_text_conditional"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    train_metrics: dict,
    valid_metrics: dict,
    config: dict,
    vocabulary: Vocabulary,
    best_valid_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "train_metrics": (
                train_metrics
            ),
            "valid_metrics": (
                valid_metrics
            ),
            "best_valid_loss": (
                best_valid_loss
            ),
            "config": config,
            "vocabulary": (
                vocabulary.to_dict()
            ),
        },
        path,
    )


def main():
    config = load_config()

    seed = int(
        config[
            "training"
        ][
            "seed"
        ]
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
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

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ---------------------------------
    # Vocabulary
    # ---------------------------------

    vocabulary_path = Path(
        config[
            "text"
        ][
            "vocabulary_path"
        ]
    )

    vocabulary = Vocabulary.load(
        vocabulary_path
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
        len(
            vocabulary
        ),
    )

    print(
        "Max text length:",
        max_length,
    )

    # ---------------------------------
    # Dataset
    # ---------------------------------

    train_dataset = TextProductDataset(
        metadata_path=Path(
            config[
                "data"
            ][
                "train_metadata"
            ]
        ),
        vocabulary=vocabulary,
        max_length=max_length,
    )

    valid_dataset = TextProductDataset(
        metadata_path=Path(
            config[
                "data"
            ][
                "valid_metadata"
            ]
        ),
        vocabulary=vocabulary,
        max_length=max_length,
    )

    print(
        "Train samples:",
        len(
            train_dataset
        ),
    )

    print(
        "Valid samples:",
        len(
            valid_dataset
        ),
    )

    batch_size = int(
        config[
            "training"
        ][
            "batch_size"
        ]
    )

    num_workers = int(
        config[
            "training"
        ][
            "num_workers"
        ]
    )

    # DataLoader shuffle RNG stays CPU.
    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
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
    # Optimizer
    # ---------------------------------

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

    # Training RNG controls:
    # - prompt dropout
    # - random timestep
    # - Gaussian noise
    train_generator = torch.Generator(
        device=device.type
    )

    train_generator.manual_seed(
        seed + 1000
    )

    epochs = int(
        config[
            "training"
        ][
            "epochs"
        ]
    )

    prompt_dropout = float(
        config[
            "conditioning"
        ][
            "prompt_dropout"
        ]
    )

    checkpoint_interval = int(
        config[
            "training"
        ][
            "checkpoint_interval"
        ]
    )

    print(
        "Prompt dropout:",
        prompt_dropout,
    )

    # ---------------------------------
    # Output
    # ---------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_valid_loss = float(
        "inf"
    )

    print()
    print(
        "Starting Phase 3 "
        "full text-conditioned training..."
    )

    # ---------------------------------
    # Training
    # ---------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_metrics = (
            train_text_epoch(
                model=model,
                scheduler=scheduler,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                prompt_dropout=(
                    prompt_dropout
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
                generator=train_generator,
            )
        )

        # Same validation stochastic
        # sequence every epoch.
        valid_generator = (
            torch.Generator(
                device=device.type
            )
        )

        valid_generator.manual_seed(
            seed + 2000
        )

        valid_metrics = (
            validate_text_epoch(
                model=model,
                scheduler=scheduler,
                dataloader=valid_loader,
                device=device,
                generator=valid_generator,
            )
        )

        train_loss = float(
            train_metrics[
                "loss"
            ]
        )

        valid_loss = float(
            valid_metrics[
                "loss"
            ]
        )

        dropped_fraction = float(
            train_metrics[
                "dropped_fraction"
            ]
        )

        print(
            f"epoch={epoch:03d}/"
            f"{epochs} "
            f"train="
            f"{train_loss:.6f} "
            f"valid="
            f"{valid_loss:.6f} "
            f"dropout="
            f"{dropped_fraction:.3f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": (
                train_loss
            ),
            "valid_loss": (
                valid_loss
            ),
            "dropped_fraction": (
                dropped_fraction
            ),
        }

        history.append(
            record
        )

        # ---------------------------------
        # Best checkpoint
        # ---------------------------------

        if (
            valid_loss
            < best_valid_loss
        ):
            best_valid_loss = (
                valid_loss
            )

            save_checkpoint(
                path=(
                    OUTPUT_ROOT
                    / "best.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=(
                    train_metrics
                ),
                valid_metrics=(
                    valid_metrics
                ),
                config=config,
                vocabulary=vocabulary,
                best_valid_loss=(
                    best_valid_loss
                ),
            )

            print(
                "  -> saved best.pt"
            )

        # ---------------------------------
        # Periodic checkpoint
        # ---------------------------------

        if (
            epoch
            % checkpoint_interval
            == 0
        ):
            save_checkpoint(
                path=(
                    OUTPUT_ROOT
                    / (
                        "checkpoint_"
                        f"epoch_{epoch:03d}.pt"
                    )
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=(
                    train_metrics
                ),
                valid_metrics=(
                    valid_metrics
                ),
                config=config,
                vocabulary=vocabulary,
                best_valid_loss=(
                    best_valid_loss
                ),
            )

        # ---------------------------------
        # History
        # ---------------------------------

        with (
            OUTPUT_ROOT
            / "history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
            )

    # ---------------------------------
    # Final checkpoint
    # ---------------------------------

    save_checkpoint(
        path=(
            OUTPUT_ROOT
            / "last.pt"
        ),
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        train_metrics=(
            train_metrics
        ),
        valid_metrics=(
            valid_metrics
        ),
        config=config,
        vocabulary=vocabulary,
        best_valid_loss=(
            best_valid_loss
        ),
    )

    best_record = min(
        history,
        key=lambda item: (
            item[
                "valid_loss"
            ]
        ),
    )

    summary = {
        "epochs": epochs,
        "train_samples": len(
            train_dataset
        ),
        "valid_samples": len(
            valid_dataset
        ),
        "vocabulary_size": len(
            vocabulary
        ),
        "max_length": (
            max_length
        ),
        "parameter_count": (
            parameter_count
        ),
        "prompt_dropout": (
            prompt_dropout
        ),
        "best_epoch": int(
            best_record[
                "epoch"
            ]
        ),
        "best_valid_loss": float(
            best_record[
                "valid_loss"
            ]
        ),
        "final_train_loss": float(
            train_metrics[
                "loss"
            ]
        ),
        "final_valid_loss": float(
            valid_metrics[
                "loss"
            ]
        ),
    }

    with (
        OUTPUT_ROOT
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 80)

    print(
        "Training complete"
    )

    print(
        "Best epoch:",
        summary[
            "best_epoch"
        ],
    )

    print(
        "Best valid loss:",
        summary[
            "best_valid_loss"
        ],
    )

    print(
        "Output:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()