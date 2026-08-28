import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROCESSED_ROOT = Path(
    "data/processed/products_64"
)

OUTPUT_ROOT = Path(
    "outputs/phase0_data"
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

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def create_contact_sheet(
    split: str,
    max_images: int | None,
) -> None:
    metadata_path = (
        PROCESSED_ROOT
        / f"{split}.jsonl"
    )

    records = load_jsonl(
        metadata_path
    )

    if max_images is not None:
        records = records[:max_images]

    if not records:
        raise RuntimeError(
            f"No records found for split={split}"
        )

    num_images = len(records)

    num_columns = 8

    num_rows = math.ceil(
        num_images / num_columns
    )

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(
            num_columns * 2,
            num_rows * 2.3,
        ),
    )

    axes = axes.reshape(-1)

    for ax, record in zip(
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
            image = image.convert("RGB")

            ax.imshow(
                image,
                interpolation="nearest",
            )

        ax.set_title(
            f"img={record['image_id']}\n"
            f"ann={record['annotation_id']}",
            fontsize=7,
        )

        ax.axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / f"contact_sheet_{split}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"{split}: {num_images} images "
        f"-> {output_path}"
    )


def main():
    create_contact_sheet(
        split="train",
        max_images=32,
    )

    create_contact_sheet(
        split="valid",
        max_images=None,
    )

    create_contact_sheet(
        split="test",
        max_images=None,
    )


if __name__ == "__main__":
    main()