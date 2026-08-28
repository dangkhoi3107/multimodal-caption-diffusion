import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.product_dataset import (
    ProductImageDataset,
)


def create_test_dataset(tmp_path):
    train_dir = tmp_path / "train"

    train_dir.mkdir()

    image = np.full(
        (64, 64, 3),
        128,
        dtype=np.uint8,
    )

    image_path = (
        train_dir / "sample.png"
    )

    Image.fromarray(image).save(
        image_path
    )

    record = {
        "file_name": "train/sample.png",
        "class_id": 0,
        "class_name": "lifebuoy",
        "image_size": 64,
    }

    metadata_path = (
        tmp_path / "train.jsonl"
    )

    metadata_path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    return metadata_path


def test_product_dataset_length(
    tmp_path,
):
    metadata_path = (
        create_test_dataset(tmp_path)
    )

    dataset = ProductImageDataset(
        metadata_path
    )

    assert len(dataset) == 1


def test_product_dataset_item(
    tmp_path,
):
    metadata_path = (
        create_test_dataset(tmp_path)
    )

    dataset = ProductImageDataset(
        metadata_path
    )

    item = dataset[0]

    image = item["image"]

    assert image.shape == (
        3,
        64,
        64,
    )

    assert image.dtype == torch.float32

    assert torch.isfinite(
        image
    ).all()

    assert image.min() >= -1.0
    assert image.max() <= 1.0

    assert item["class_id"].dtype == (
        torch.long
    )

    assert item["class_id"].item() == 0


def test_product_dataset_dataloader(
    tmp_path,
):
    metadata_path = (
        create_test_dataset(tmp_path)
    )

    dataset = ProductImageDataset(
        metadata_path
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    batch = next(
        iter(loader)
    )

    assert batch["image"].shape == (
        1,
        3,
        64,
        64,
    )

    assert batch[
        "class_id"
    ].shape == (1,)