from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.caption_dataset import CaptionDataset
from src.text.tokenizer import decode
from src.text.vocabulary import Vocabulary


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

    vocabulary_path = Path(
        config[
            "text"
        ][
            "vocabulary_path"
        ]
    )

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            "Vocabulary not found: "
            f"{vocabulary_path}\n"
            "Run the Phase 3 vocabulary builder first."
        )

    vocabulary = Vocabulary.load(
        vocabulary_path
    )

    sequence_length = int(
        config[
            "text"
        ][
            "sequence_length"
        ]
    )

    split_paths = {
        "train": Path(
            config[
                "data"
            ][
                "train_metadata"
            ]
        ),
        "valid": Path(
            config[
                "data"
            ][
                "valid_metadata"
            ]
        ),
        "test": Path(
            config[
                "data"
            ][
                "test_metadata"
            ]
        ),
    }

    print(
        "Vocabulary size:",
        len(
            vocabulary
        ),
    )

    print(
        "Decoder sequence length:",
        sequence_length,
    )

    datasets = {}

    for split, metadata_path in (
        split_paths.items()
    ):
        dataset = CaptionDataset(
            metadata_path=metadata_path,
            vocabulary=vocabulary,
            sequence_length=sequence_length,
        )

        datasets[
            split
        ] = dataset

        print(
            f"{split}: "
            f"{len(dataset)} samples"
        )

    train_loader = DataLoader(
        datasets[
            "train"
        ],
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(
            train_loader
        )
    )

    print()
    print(
        "Batch image:",
        tuple(
            batch[
                "image"
            ].shape
        ),
        batch[
            "image"
        ].dtype,
    )

    print(
        "Batch input_ids:",
        tuple(
            batch[
                "input_ids"
            ].shape
        ),
        batch[
            "input_ids"
        ].dtype,
    )

    print(
        "Batch target_ids:",
        tuple(
            batch[
                "target_ids"
            ].shape
        ),
        batch[
            "target_ids"
        ].dtype,
    )

    print(
        "Batch padding_mask:",
        tuple(
            batch[
                "padding_mask"
            ].shape
        ),
        batch[
            "padding_mask"
        ].dtype,
    )

    print(
        "Batch target_mask:",
        tuple(
            batch[
                "target_mask"
            ].shape
        ),
        batch[
            "target_mask"
        ].dtype,
    )

    if batch[
        "image"
    ].shape[1:] != (
        3,
        64,
        64,
    ):
        raise AssertionError(
            "Unexpected image shape"
        )

    if batch[
        "input_ids"
    ].shape[1] != (
        sequence_length
    ):
        raise AssertionError(
            "Unexpected input sequence length"
        )

    if batch[
        "target_ids"
    ].shape[1] != (
        sequence_length
    ):
        raise AssertionError(
            "Unexpected target sequence length"
        )

    if batch[
        "padding_mask"
    ].dtype != torch.bool:
        raise AssertionError(
            "padding_mask must be bool"
        )

    if batch[
        "target_mask"
    ].dtype != torch.bool:
        raise AssertionError(
            "target_mask must be bool"
        )

    print()
    print(
        "Teacher-forcing examples:"
    )

    for index in range(
        min(
            4,
            batch[
                "input_ids"
            ].shape[
                0
            ],
        )
    ):
        caption = batch[
            "caption"
        ][
            index
        ]

        decoded_input = decode(
            batch[
                "input_ids"
            ][
                index
            ],
            vocabulary=vocabulary,
        )

        decoded_target = decode(
            batch[
                "target_ids"
            ][
                index
            ],
            vocabulary=vocabulary,
        )

        print(
            f"[{index}] caption: "
            f"{caption}"
        )

        print(
            f"    input : "
            f"{decoded_input}"
        )

        print(
            f"    target: "
            f"{decoded_target}"
        )

    print()
    print(
        "Phase 4 dataset smoke test: PASS"
    )


if __name__ == "__main__":
    main()
