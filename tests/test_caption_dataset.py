from __future__ import annotations

import json

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.caption_dataset import CaptionDataset
from src.text.vocabulary import build_vocabulary


def make_fixture(
    tmp_path,
):
    image = Image.new(
        "RGB",
        (
            64,
            64,
        ),
        color=(
            255,
            0,
            0,
        ),
    )

    image_path = (
        tmp_path
        / "sample.png"
    )

    image.save(
        image_path
    )

    metadata_path = (
        tmp_path
        / "train_phase4.jsonl"
    )

    record = {
        "file_name": (
            "sample.png"
        ),
        "class_id": 2,
        "class_name": (
            "lifebuoy_handwash"
        ),
        "image_size": 64,
        "caption": (
            "a red pouch"
        ),
    }

    metadata_path.write_text(
        json.dumps(
            record
        )
        + "\n",
        encoding="utf-8",
    )

    vocabulary = build_vocabulary(
        [
            [
                "a",
                "red",
                "pouch",
            ]
        ]
    )

    return (
        metadata_path,
        vocabulary,
    )


def test_caption_dataset_contract(
    tmp_path,
):
    (
        metadata_path,
        vocabulary,
    ) = make_fixture(
        tmp_path
    )

    dataset = CaptionDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        sequence_length=6,
    )

    item = dataset[
        0
    ]

    assert item[
        "image"
    ].shape == (
        3,
        64,
        64,
    )

    assert (
        item[
            "image"
        ].dtype
        == torch.float32
    )

    assert (
        item[
            "input_ids"
        ].shape
        == (
            6,
        )
    )

    assert (
        item[
            "target_ids"
        ].shape
        == (
            6,
        )
    )

    assert (
        item[
            "padding_mask"
        ].shape
        == (
            6,
        )
    )

    assert (
        item[
            "target_mask"
        ].shape
        == (
            6,
        )
    )

    assert (
        item[
            "input_ids"
        ].dtype
        == torch.long
    )

    assert (
        item[
            "target_ids"
        ].dtype
        == torch.long
    )

    assert (
        item[
            "padding_mask"
        ].dtype
        == torch.bool
    )

    assert (
        item[
            "target_mask"
        ].dtype
        == torch.bool
    )

    assert (
        item[
            "caption"
        ]
        == "a red pouch"
    )


def test_teacher_forcing_shift(
    tmp_path,
):
    (
        metadata_path,
        vocabulary,
    ) = make_fixture(
        tmp_path
    )

    dataset = CaptionDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        sequence_length=6,
    )

    item = dataset[
        0
    ]

    expected_input = torch.tensor(
        [
            vocabulary.bos_id,
            vocabulary.token_id(
                "a"
            ),
            vocabulary.token_id(
                "red"
            ),
            vocabulary.token_id(
                "pouch"
            ),
            vocabulary.eos_id,
            vocabulary.pad_id,
        ],
        dtype=torch.long,
    )

    expected_target = torch.tensor(
        [
            vocabulary.token_id(
                "a"
            ),
            vocabulary.token_id(
                "red"
            ),
            vocabulary.token_id(
                "pouch"
            ),
            vocabulary.eos_id,
            vocabulary.pad_id,
            vocabulary.pad_id,
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        item[
            "input_ids"
        ],
        expected_input,
    )

    torch.testing.assert_close(
        item[
            "target_ids"
        ],
        expected_target,
    )


def test_masks_are_correct(
    tmp_path,
):
    (
        metadata_path,
        vocabulary,
    ) = make_fixture(
        tmp_path
    )

    dataset = CaptionDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        sequence_length=6,
    )

    item = dataset[
        0
    ]

    expected_input_mask = (
        item[
            "input_ids"
        ]
        != vocabulary.pad_id
    )

    expected_target_mask = (
        item[
            "target_ids"
        ]
        != vocabulary.pad_id
    )

    torch.testing.assert_close(
        item[
            "padding_mask"
        ],
        expected_input_mask,
    )

    torch.testing.assert_close(
        item[
            "target_mask"
        ],
        expected_target_mask,
    )


def test_image_range(
    tmp_path,
):
    (
        metadata_path,
        vocabulary,
    ) = make_fixture(
        tmp_path
    )

    dataset = CaptionDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        sequence_length=6,
    )

    image = dataset[
        0
    ][
        "image"
    ]

    assert (
        image.min().item()
        >= -1.0
    )

    assert (
        image.max().item()
        <= 1.0
    )

    assert torch.isfinite(
        image
    ).all()


def test_dataloader_batch_shapes(
    tmp_path,
):
    (
        metadata_path,
        vocabulary,
    ) = make_fixture(
        tmp_path
    )

    dataset = CaptionDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        sequence_length=6,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    batch = next(
        iter(
            loader
        )
    )

    assert batch[
        "image"
    ].shape == (
        1,
        3,
        64,
        64,
    )

    assert batch[
        "input_ids"
    ].shape == (
        1,
        6,
    )

    assert batch[
        "target_ids"
    ].shape == (
        1,
        6,
    )

    assert batch[
        "padding_mask"
    ].shape == (
        1,
        6,
    )

    assert batch[
        "target_mask"
    ].shape == (
        1,
        6,
    )


def test_rejects_too_short_sequence(
    tmp_path,
):
    (
        metadata_path,
        vocabulary,
    ) = make_fixture(
        tmp_path
    )

    try:
        CaptionDataset(
            metadata_path=metadata_path,
            vocabulary=vocabulary,
            sequence_length=1,
        )
    except ValueError:
        return

    raise AssertionError(
        "sequence_length=1 should fail"
    )
