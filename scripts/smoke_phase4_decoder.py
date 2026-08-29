from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.captioning.decoder import (
    CaptionDecoder,
)
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

    sequence_length = int(
        config[
            "text"
        ][
            "sequence_length"
        ]
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
        sequence_length=(
            sequence_length
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

    model_dim = int(
        config[
            "model"
        ][
            "model_dim"
        ]
    )

    image_encoder = ImageEncoder(
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
        model_dim=model_dim,
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

    decoder = CaptionDecoder(
        vocab_size=len(
            vocabulary
        ),
        pad_id=(
            vocabulary.pad_id
        ),
        max_length=(
            sequence_length
        ),
        model_dim=model_dim,
        num_heads=int(
            config[
                "model"
            ][
                "decoder"
            ][
                "num_heads"
            ]
        ),
        num_layers=int(
            config[
                "model"
            ][
                "decoder"
            ][
                "num_layers"
            ]
        ),
        feedforward_dim=int(
            config[
                "model"
            ][
                "decoder"
            ][
                "feedforward_dim"
            ]
        ),
        dropout=float(
            config[
                "model"
            ][
                "decoder"
            ][
                "dropout"
            ]
        ),
    ).to(
        device
    )

    images = batch[
        "image"
    ].to(
        device
    )

    input_ids = batch[
        "input_ids"
    ].to(
        device
    )

    padding_mask = batch[
        "padding_mask"
    ].to(
        device
    )

    with torch.no_grad():
        image_tokens = (
            image_encoder(
                images
            )
        )

        (
            logits,
            self_weights,
            cross_weights,
        ) = (
            decoder.forward_with_attention(
                input_ids=input_ids,
                padding_mask=(
                    padding_mask
                ),
                image_tokens=(
                    image_tokens
                ),
            )
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
        "Image:",
        tuple(
            images.shape
        ),
    )

    print(
        "Image tokens:",
        tuple(
            image_tokens.shape
        ),
    )

    print(
        "Decoder input:",
        tuple(
            input_ids.shape
        ),
    )

    print(
        "Logits:",
        tuple(
            logits.shape
        ),
    )

    print(
        "Self-attention:",
        tuple(
            self_weights[
                0
            ].shape
        ),
    )

    print(
        "Cross-attention:",
        tuple(
            cross_weights[
                0
            ].shape
        ),
    )

    encoder_parameters = sum(
        parameter.numel()
        for parameter
        in image_encoder.parameters()
    )

    decoder_parameters = sum(
        parameter.numel()
        for parameter
        in decoder.parameters()
    )

    print(
        "Image encoder parameters:",
        f"{encoder_parameters:,}",
    )

    print(
        "Decoder parameters:",
        f"{decoder_parameters:,}",
    )

    print(
        "Total parameters:",
        f"{encoder_parameters + decoder_parameters:,}",
    )

    expected_logits_shape = (
        images.shape[
            0
        ],
        sequence_length,
        len(
            vocabulary
        ),
    )

    if logits.shape != (
        expected_logits_shape
    ):
        raise AssertionError(
            "Unexpected decoder logits shape"
        )

    if not torch.isfinite(
        logits
    ).all():
        raise AssertionError(
            "Decoder logits contain NaN/Inf"
        )

    print()
    print(
        "Phase 4 decoder smoke test: PASS"
    )


if __name__ == "__main__":
    main()
