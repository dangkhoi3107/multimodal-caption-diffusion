import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.product_dataset import ProductImageDataset


PROCESSED_ROOT = Path(
    "data/processed/products_multiclass_64"
)

OUTPUT_ROOT = Path(
    "outputs/phase2_data"
)

SEED = 42
MAX_IMAGES_PER_CLASS = 24


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def inspect_split(
    split: str,
    classes: list[dict],
) -> dict:
    metadata_path = (
        PROCESSED_ROOT
        / f"{split}.jsonl"
    )

    records = load_jsonl(
        metadata_path
    )

    class_counts = Counter(
        int(record["class_id"])
        for record in records
    )

    print()
    print("=" * 80)
    print("split:", split)
    print("records:", len(records))

    for class_info in classes:
        class_id = int(
            class_info["class_id"]
        )

        class_name = str(
            class_info["class_name"]
        )

        print(
            f"class_id={class_id} "
            f"count={class_counts[class_id]} "
            f"name={class_name}"
        )

    return {
        "num_records": len(records),
        "class_counts": {
            str(class_id): count
            for class_id, count
            in sorted(
                class_counts.items()
            )
        },
    }


def inspect_dataloader() -> dict:
    dataset = ProductImageDataset(
        PROCESSED_ROOT
        / "train.jsonl"
    )

    loader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    images = batch["image"]
    class_ids = batch["class_id"]

    if images.shape[1:] != (
        3,
        64,
        64,
    ):
        raise RuntimeError(
            f"Unexpected batch shape: "
            f"{tuple(images.shape)}"
        )

    if images.dtype != torch.float32:
        raise RuntimeError(
            f"Unexpected dtype: "
            f"{images.dtype}"
        )

    if not torch.isfinite(
        images
    ).all():
        raise RuntimeError(
            "Batch contains NaN/Inf"
        )

    minimum = float(
        images.min().item()
    )

    maximum = float(
        images.max().item()
    )

    if (
        minimum < -1.0
        or maximum > 1.0
    ):
        raise RuntimeError(
            "Images outside [-1, 1]"
        )

    print()
    print("=" * 80)
    print("DataLoader QA")
    print(
        "batch shape:",
        tuple(images.shape),
    )
    print(
        "dtype:",
        images.dtype,
    )
    print(
        "min:",
        minimum,
    )
    print(
        "max:",
        maximum,
    )
    print(
        "class_ids:",
        class_ids.tolist(),
    )

    return {
        "batch_shape": list(
            images.shape
        ),
        "dtype": str(
            images.dtype
        ),
        "min": minimum,
        "max": maximum,
        "finite": True,
    }


def create_contact_sheet(
    split: str,
    class_id: int,
    class_name: str,
) -> Path:
    records = load_jsonl(
        PROCESSED_ROOT
        / f"{split}.jsonl"
    )

    records = [
        record
        for record in records
        if int(
            record["class_id"]
        ) == class_id
    ]

    rng = random.Random(
        SEED
        + class_id
    )

    rng.shuffle(
        records
    )

    records = records[
        :MAX_IMAGES_PER_CLASS
    ]

    if not records:
        raise RuntimeError(
            f"No records for "
            f"split={split}, "
            f"class_id={class_id}"
        )

    num_columns = 6

    num_rows = math.ceil(
        len(records)
        / num_columns
    )

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(
            num_columns * 2,
            num_rows * 2.2,
        ),
    )

    if hasattr(
        axes,
        "reshape",
    ):
        axes = axes.reshape(-1)
    else:
        axes = [axes]

    for axis, record in zip(
        axes,
        records,
    ):
        image_path = (
            PROCESSED_ROOT
            / record["file_name"]
        )

        with Image.open(
            image_path
        ) as image:
            axis.imshow(
                image.convert("RGB")
            )

        axis.set_title(
            f"img={record['image_id']}\n"
            f"ann={record['annotation_id']}",
            fontsize=7,
        )

        axis.axis("off")

    for axis in axes[
        len(records):
    ]:
        axis.axis("off")

    fig.suptitle(
        f"{split} | "
        f"class {class_id} | "
        f"{class_name}",
        fontsize=10,
    )

    plt.tight_layout()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / (
            f"contact_sheet_"
            f"{split}_"
            f"class_{class_id}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        output_path,
    )

    return output_path


def main():
    classes = load_json(
        PROCESSED_ROOT
        / "classes.json"
    )

    print(
        "Number of classes:",
        len(classes),
    )

    if len(classes) != 3:
        raise RuntimeError(
            "Expected exactly 3 classes"
        )

    qa_report = {
        "classes": classes,
        "splits": {},
    }

    for split in [
        "train",
        "valid",
        "test",
    ]:
        qa_report[
            "splits"
        ][split] = inspect_split(
            split=split,
            classes=classes,
        )

    qa_report[
        "dataloader"
    ] = inspect_dataloader()

    contact_sheets = []

    for split in [
        "train",
        "valid",
        "test",
    ]:
        for class_info in classes:
            path = create_contact_sheet(
                split=split,
                class_id=int(
                    class_info[
                        "class_id"
                    ]
                ),
                class_name=str(
                    class_info[
                        "class_name"
                    ]
                ),
            )

            contact_sheets.append(
                str(path)
            )

    qa_report[
        "contact_sheets"
    ] = contact_sheets

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        OUTPUT_ROOT
        / "qa_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            qa_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print(
        "QA report:",
        report_path,
    )


if __name__ == "__main__":
    main()