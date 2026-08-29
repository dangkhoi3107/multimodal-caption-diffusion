import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.text_product_dataset import (
    TextProductDataset,
)
from src.text.tokenizer import (
    tokenize,
)
from src.text.vocabulary import (
    build_vocabulary,
)


def make_fixture(
    tmp_path: Path,
):
    image_path = (
        tmp_path
        / "sample.png"
    )

    image = np.full(
        (64, 64, 3),
        200,
        dtype=np.uint8,
    )

    Image.fromarray(
        image
    ).save(
        image_path
    )

    caption = (
        "a red lifebuoy handwash pouch"
    )

    metadata_path = (
        tmp_path
        / "train_phase3.jsonl"
    )

    record = {
        "file_name": "sample.png",
        "class_id": 2,
        "class_name": (
            "lifebuoy_handwash_"
            "vitamin_protection_400g"
        ),
        "image_size": 64,
        "caption": caption,
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
            tokenize(
                caption
            )
        ]
    )

    return (
        metadata_path,
        vocabulary,
        caption,
    )


def test_text_dataset_fields(
    tmp_path: Path,
):
    (
        metadata_path,
        vocabulary,
        caption,
    ) = make_fixture(
        tmp_path
    )

    dataset = TextProductDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        max_length=10,
    )

    item = dataset[0]

    assert item[
        "image"
    ].shape == (
        3,
        64,
        64,
    )

    assert (
        item["caption"]
        == caption
    )

    assert item[
        "token_ids"
    ].shape == (
        10,
    )

    assert (
        item[
            "token_ids"
        ].dtype
        == torch.long
    )

    assert item[
        "padding_mask"
    ].shape == (
        10,
    )

    assert (
        item[
            "padding_mask"
        ].dtype
        == torch.bool
    )


def test_text_dataset_bos_eos(
    tmp_path: Path,
):
    (
        metadata_path,
        vocabulary,
        _,
    ) = make_fixture(
        tmp_path
    )

    dataset = TextProductDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        max_length=10,
    )

    token_ids = dataset[
        0
    ][
        "token_ids"
    ]

    assert (
        token_ids[0].item()
        == vocabulary.bos_id
    )

    assert (
        vocabulary.eos_id
        in token_ids.tolist()
    )


def test_text_dataset_image_range(
    tmp_path: Path,
):
    (
        metadata_path,
        vocabulary,
        _,
    ) = make_fixture(
        tmp_path
    )

    dataset = TextProductDataset(
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        max_length=10,
    )

    image = dataset[
        0
    ][
        "image"
    ]

    assert torch.isfinite(
        image
    ).all()

    assert (
        image.min().item()
        >= -1.0
    )

    assert (
        image.max().item()
        <= 1.0
    )