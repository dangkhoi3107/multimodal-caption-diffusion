import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from src.data.product_dataset import ProductImageDataset


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

CHECKPOINT_PATH = Path(
    "checkpoints/phase1_unconditional/best.pt"
)

OUTPUT_PATH = Path(
    "outputs/phase1_unconditional/run_metadata.json"
)


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def main() -> None:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    train_dataset = ProductImageDataset(
        Path(
            config["data"]["train_metadata"]
        )
    )

    valid_dataset = ProductImageDataset(
        Path(
            config["data"]["valid_metadata"]
        )
    )

    cuda_available = (
        torch.cuda.is_available()
    )

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)

        gpu_properties = (
            torch.cuda.get_device_properties(0)
        )

        gpu_memory_gb = (
            gpu_properties.total_memory
            / (1024 ** 3)
        )
    else:
        gpu_name = None
        gpu_memory_gb = None

    metadata = {
        "phase": "Phase 1 - Unconditional Pixel DDPM",

        "recorded_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "git": {
            "commit": git_value(
                "rev-parse",
                "HEAD",
            ),
            "branch": git_value(
                "branch",
                "--show-current",
            ),
            "status": git_value(
                "status",
                "--short",
            ),
        },

        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if cuda_available
                else None
            ),
        },

        "hardware": {
            "gpu": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "cpu": platform.processor(),
        },

        "dataset": {
            "train_images": len(
                train_dataset
            ),
            "valid_images": len(
                valid_dataset
            ),
            "image_size": config[
                "data"
            ]["image_size"],
        },

        "model": {
            "base_channels": config[
                "model"
            ]["base_channels"],
            "parameter_count": 4603587,
        },

        "diffusion": {
            "num_timesteps": config[
                "diffusion"
            ]["num_timesteps"],
            "beta_start": config[
                "diffusion"
            ]["beta_start"],
            "beta_end": config[
                "diffusion"
            ]["beta_end"],
        },

        "training": {
            "seed": config[
                "training"
            ]["seed"],
            "batch_size": config[
                "training"
            ]["batch_size"],
            "learning_rate": config[
                "training"
            ]["learning_rate"],
            "configured_epochs": config[
                "training"
            ]["epochs"],

            # Nếu lúc train Kaggle chưa đo thời gian,
            # để null thay vì tự đoán.
            "duration_seconds": None,
            "duration_note": (
                "Original Kaggle baseline duration "
                "was not recorded automatically."
            ),
        },

        "best_checkpoint": {
            "path": str(
                CHECKPOINT_PATH
            ),
            "epoch": checkpoint.get(
                "epoch"
            ),
            "train_loss": checkpoint.get(
                "train_loss"
            ),
            "valid_loss": checkpoint.get(
                "valid_loss"
            ),
        },

        "sampling_evidence": {
            "seed": 342,
            "final_min": -0.9998340606689453,
            "final_max": 0.9998340606689453,
            "final_mean": 0.4951973855495453,
            "final_std": 0.5756363868713379,
            "fraction_below_minus_one": 0.0,
            "fraction_above_one": 0.0,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print()
    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()