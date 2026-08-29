from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.captioning.image_encoder import (
    ImageEncoder,
)
from src.data.caption_dataset import (
    CaptionDataset,
)
from src.text.vocabulary import (
    Vocabulary,
)


CONFIG_PATH = Path(
    "configs/phase4_captioning.yaml"
)


def main():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    vocabulary = Vocabulary.load(
        Path(
            config[
                "text"
            ][
                "vocabulary_path"
            ]
        )
    )

    dataset = CaptionDataset(
        metadata_path=Path(
            config[
                "data"
            ][
                "train_metadata"
            ]
        ),
        vocabulary=vocabulary,
        sequence_length=int(
            config[
                "text"
            ][
                "sequence_length"
            ]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(
            loader
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ImageEncoder(
        in_channels=int(
            config[
                "model"
            ][
                "image_encoder"
            ][
                "in_channels"
            ]
        ),
        base_channels=int(
            config[
                "model"
            ][
                "image_encoder"
            ][
                "base_channels"
            ]
        ),
        model_dim=int(
            config[
                "model"
            ][
                "model_dim"
            ]
        ),
        image_size=int(
            config[
                "data"
            ][
                "image_size"
            ]
        ),
    ).to(
        device
    )

    images = batch[
        "image"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        tokens = model(
            images
        )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
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

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    print(
        "Input image:",
        tuple(
            images.shape
        ),
    )

    print(
        "Image tokens:",
        tuple(
            tokens.shape
        ),
    )

    print(
        "Grid size:",
        model.grid_size,
    )

    print(
        "Num image tokens:",
        model.num_tokens,
    )

    print(
        "Token min/max:",
        float(
            tokens.min().item()
        ),
        float(
            tokens.max().item()
        ),
    )

    if tokens.shape != (
        images.shape[
            0
        ],
        model.num_tokens,
        int(
            config[
                "model"
            ][
                "model_dim"
            ]
        ),
    ):
        raise AssertionError(
            "Unexpected image-token shape"
        )

    if not torch.isfinite(
        tokens
    ).all():
        raise AssertionError(
            "Image tokens contain NaN/Inf"
        )

    print()
    print(
        "Phase 4 image encoder smoke test: PASS"
    )


if __name__ == "__main__":
    main()
