import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from src.data.product_dataset import (
    ProductImageDataset,
)


PROCESSED_ROOT = Path(
    "data/processed/products_64"
)

OUTPUT_ROOT = Path(
    "outputs/phase0_data"
)

SPLITS = (
    "train",
    "valid",
    "test",
)


def load_jsonl(
    path: Path,
) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def inspect_split(
    split: str,
) -> dict:
    metadata_path = (
        PROCESSED_ROOT
        / f"{split}.jsonl"
    )

    records = load_jsonl(
        metadata_path
    )

    invalid_images = []
    image_hashes = {}

    min_pixel = 255
    max_pixel = 0

    for record in records:
        image_path = (
            PROCESSED_ROOT
            / record["file_name"]
        )

        try:
            with Image.open(
                image_path
            ) as image:
                image = image.convert("RGB")

                array = np.asarray(
                    image,
                    dtype=np.uint8,
                )

            if array.shape != (
                64,
                64,
                3,
            ):
                invalid_images.append(
                    {
                        "file_name": (
                            record["file_name"]
                        ),
                        "reason": (
                            f"shape={array.shape}"
                        ),
                    }
                )

            if array.dtype != np.uint8:
                invalid_images.append(
                    {
                        "file_name": (
                            record["file_name"]
                        ),
                        "reason": (
                            f"dtype={array.dtype}"
                        ),
                    }
                )

            min_pixel = min(
                min_pixel,
                int(array.min()),
            )

            max_pixel = max(
                max_pixel,
                int(array.max()),
            )

            digest = sha256_file(
                image_path
            )

            image_hashes.setdefault(
                digest,
                [],
            ).append(
                record["file_name"]
            )

        except Exception as error:
            invalid_images.append(
                {
                    "file_name": (
                        record["file_name"]
                    ),
                    "reason": str(error),
                }
            )

    duplicate_groups = [
        files
        for files in image_hashes.values()
        if len(files) > 1
    ]

    class_ids = sorted(
        {
            int(record["class_id"])
            for record in records
        }
    )

    return {
        "metadata_records": len(records),
        "invalid_images": invalid_images,
        "invalid_count": len(
            invalid_images
        ),
        "uint8_min": min_pixel,
        "uint8_max": max_pixel,
        "class_ids": class_ids,
        "exact_duplicate_groups": (
            duplicate_groups
        ),
    }


def inspect_dataloader() -> dict:
    dataset = ProductImageDataset(
        PROCESSED_ROOT / "train.jsonl"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    batch = next(
        iter(loader)
    )

    images = batch["image"]

    return {
        "dataset_size": len(dataset),
        "batch_shape": list(
            images.shape
        ),
        "dtype": str(
            images.dtype
        ),
        "min": float(
            images.min().item()
        ),
        "max": float(
            images.max().item()
        ),
        "class_ids": (
            batch["class_id"]
            .tolist()
        ),
    }


def main():
    report = {
        "splits": {},
    }

    for split in SPLITS:
        report["splits"][split] = (
            inspect_split(split)
        )

    report["dataloader"] = (
        inspect_dataloader()
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "qa_report.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"QA report saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()