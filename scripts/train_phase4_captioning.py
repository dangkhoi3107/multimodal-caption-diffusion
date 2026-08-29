from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.captioning.model import CaptionModel
from src.captioning.training import (
    train_caption_epoch,
    validate_caption_epoch,
)
from src.data.caption_dataset import CaptionDataset
from src.text.vocabulary import Vocabulary


CONFIG_PATH = Path(
    "configs/phase4_captioning.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/phase4_captioning"
)


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: dict,
    vocabulary: Vocabulary,
) -> CaptionModel:
    return CaptionModel(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        image_size=int(
            config["data"]["image_size"]
        ),
        in_channels=int(
            config["model"]["image_encoder"]["in_channels"]
        ),
        base_channels=int(
            config["model"]["image_encoder"]["base_channels"]
        ),
        model_dim=int(
            config["model"]["model_dim"]
        ),
        max_length=int(
            config["text"]["sequence_length"]
        ),
        num_heads=int(
            config["model"]["decoder"]["num_heads"]
        ),
        num_layers=int(
            config["model"]["decoder"]["num_layers"]
        ),
        feedforward_dim=int(
            config["model"]["decoder"]["feedforward_dim"]
        ),
        dropout=float(
            config["model"]["decoder"]["dropout"]
        ),
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: CaptionModel,
    optimizer: torch.optim.Optimizer,
    train_metrics: dict,
    valid_metrics: dict,
    best_valid_loss: float,
    config: dict,
    vocabulary: Vocabulary,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "best_valid_loss": best_valid_loss,
            "config": config,
            "vocabulary": vocabulary.to_dict(),
        },
        path,
    )


def main():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    seed = int(
        config["training"]["seed"]
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    vocabulary_path = Path(
        config["text"]["vocabulary_path"]
    )

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocabulary_path}"
        )

    vocabulary = Vocabulary.load(
        vocabulary_path
    )

    sequence_length = int(
        config["text"]["sequence_length"]
    )

    train_dataset = CaptionDataset(
        metadata_path=Path(
            config["data"]["train_metadata"]
        ),
        vocabulary=vocabulary,
        sequence_length=sequence_length,
    )

    valid_dataset = CaptionDataset(
        metadata_path=Path(
            config["data"]["valid_metadata"]
        ),
        vocabulary=vocabulary,
        sequence_length=sequence_length,
    )

    batch_size = int(
        config["training"]["batch_size"]
    )

    num_workers = int(
        config["training"]["num_workers"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model = build_model(
        config,
        vocabulary,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"]["learning_rate"]
        ),
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Vocabulary size:",
        len(vocabulary),
    )
    print(
        "Train samples:",
        len(train_dataset),
    )
    print(
        "Valid samples:",
        len(valid_dataset),
    )
    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = int(
        config["training"]["epochs"]
    )

    checkpoint_interval = int(
        config["training"]["checkpoint_interval"]
    )

    best_valid_loss = float("inf")
    best_epoch = None
    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_metrics = train_caption_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=vocabulary.pad_id,
            max_grad_norm=1.0,
        )

        valid_metrics = validate_caption_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            pad_id=vocabulary.pad_id,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }

        history.append(record)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['token_accuracy']:.3f} "
            f"valid_loss={valid_metrics['loss']:.6f} "
            f"valid_acc={valid_metrics['token_accuracy']:.3f}"
        )

        if (
            valid_metrics["loss"]
            < best_valid_loss
        ):
            best_valid_loss = float(
                valid_metrics["loss"]
            )
            best_epoch = epoch

            save_checkpoint(
                OUTPUT_ROOT / "best.pt",
                epoch,
                model,
                optimizer,
                train_metrics,
                valid_metrics,
                best_valid_loss,
                config,
                vocabulary,
            )

            print(
                f"  -> best valid loss: "
                f"{best_valid_loss:.6f}"
            )

        if (
            checkpoint_interval > 0
            and epoch % checkpoint_interval == 0
        ):
            save_checkpoint(
                OUTPUT_ROOT
                / f"checkpoint_epoch_{epoch:03d}.pt",
                epoch,
                model,
                optimizer,
                train_metrics,
                valid_metrics,
                best_valid_loss,
                config,
                vocabulary,
            )

        (
            OUTPUT_ROOT
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

    save_checkpoint(
        OUTPUT_ROOT / "last.pt",
        epochs,
        model,
        optimizer,
        history[-1]["train"],
        history[-1]["valid"],
        best_valid_loss,
        config,
        vocabulary,
    )

    summary = {
        "epochs": epochs,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "vocabulary_size": len(vocabulary),
        "sequence_length": sequence_length,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "final_train": history[-1]["train"],
        "final_valid": history[-1]["valid"],
    }

    (
        OUTPUT_ROOT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Training complete")
    print("Best epoch:", best_epoch)
    print(
        "Best valid loss:",
        best_valid_loss,
    )
    print("Output:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
