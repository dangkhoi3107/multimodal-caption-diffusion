import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.product_dataset import (
    ProductImageDataset,
)
from src.diffusion.conditional_trainer import (
    train_conditional_epoch,
    validate_conditional_epoch,
)
from src.diffusion.conditional_unet import (
    ConditionalUNet,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


CONFIG_PATH = Path(
    "configs/phase2_class_conditional.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/phase2_conditional"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    train_metrics: dict,
    valid_metrics: dict,
    config: dict,
    best_valid_loss: float,
):
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
            "num_classes": (
                model.num_classes
            ),
            "null_class_id": (
                model.null_class_id
            ),
            "config": config,
        },
        path,
    )


def main():
    config = load_config()

    seed = int(
        config["training"]["seed"]
    )

    random.seed(seed)
    torch.manual_seed(seed)

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

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------
    # Dataset
    # ---------------------------------

    train_dataset = ProductImageDataset(
        Path(
            config["data"][
                "train_metadata"
            ]
        )
    )

    valid_dataset = ProductImageDataset(
        Path(
            config["data"][
                "valid_metadata"
            ]
        )
    )

    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Valid samples:",
        len(valid_dataset),
    )

    batch_size = int(
        config["training"][
            "batch_size"
        ]
    )

    num_workers = int(
        config["training"][
            "num_workers"
        ]
    )

    # DataLoader shuffle RNG stays on CPU.
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
    # Model
    # ---------------------------------

    num_classes = int(
        config["data"][
            "num_classes"
        ]
    )

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

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    print(
        "Num classes:",
        num_classes,
    )

    print(
        "Null class ID:",
        model.null_class_id,
    )

    # ---------------------------------
    # Optimizer
    # ---------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"][
                "learning_rate"
            ]
        ),
    )

    # RNG used for training diffusion
    # timestep/noise/dropout.
    train_generator = torch.Generator(
        device=device.type
    )

    train_generator.manual_seed(
        seed + 1000
    )

    epochs = int(
        config["training"][
            "epochs"
        ]
    )

    condition_dropout = float(
        config["training"][
            "condition_dropout"
        ]
    )

    checkpoint_interval = int(
        config["training"][
            "checkpoint_interval"
        ]
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
        "Starting Phase 2 "
        "full training..."
    )

    # ---------------------------------
    # Training loop
    # ---------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_metrics = (
            train_conditional_epoch(
                model=model,
                scheduler=scheduler,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                condition_dropout=(
                    condition_dropout
                ),
                null_class_id=(
                    model.null_class_id
                ),
                generator=train_generator,
            )
        )

        # Reset validation RNG every
        # epoch so validation uses the
        # same t/noise sequence.
        valid_generator = (
            torch.Generator(
                device=device.type
            )
        )

        valid_generator.manual_seed(
            seed + 2000
        )

        valid_metrics = (
            validate_conditional_epoch(
                model=model,
                scheduler=scheduler,
                dataloader=valid_loader,
                device=device,
                generator=valid_generator,
            )
        )

        train_loss = float(
            train_metrics["loss"]
        )

        valid_loss = float(
            valid_metrics["loss"]
        )

        print(
            f"epoch={epoch:03d}/{epochs} "
            f"train={train_loss:.6f} "
            f"valid={valid_loss:.6f} "
            f"dropout="
            f"{train_metrics['dropped_fraction']:.3f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "dropped_fraction": float(
                train_metrics[
                    "dropped_fraction"
                ]
            ),
        }

        history.append(
            record
        )

        # -----------------------------
        # Best checkpoint
        # -----------------------------

        if valid_loss < (
            best_valid_loss
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
                best_valid_loss=(
                    best_valid_loss
                ),
            )

            print(
                "  -> saved best.pt"
            )

        # -----------------------------
        # Periodic checkpoint
        # -----------------------------

        if (
            epoch
            % checkpoint_interval
            == 0
        ):
            save_checkpoint(
                path=(
                    OUTPUT_ROOT
                    / (
                        f"checkpoint_"
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
                best_valid_loss=(
                    best_valid_loss
                ),
            )

        # Always update history
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
    # Last checkpoint
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
        best_valid_loss=(
            best_valid_loss
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
        "num_classes": (
            num_classes
        ),
        "null_class_id": (
            model.null_class_id
        ),
        "condition_dropout": (
            condition_dropout
        ),
        "best_valid_loss": (
            best_valid_loss
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
        "Best valid loss:",
        best_valid_loss,
    )
    print(
        "Output:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()