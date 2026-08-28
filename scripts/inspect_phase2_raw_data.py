import json
from collections import Counter
from pathlib import Path


RAW_ROOT = Path("data/raw/products")

DATASETS = [
    "dove_body_serum_glow_recharge_547ml",
    "dove_deodorant_niacinamide_omega_40ml",
    "lifebuoy_handwash_vitamin_protection_400g",
]

SPLITS = [
    "train",
    "valid",
    "test",
]


def inspect_split(
    dataset_name: str,
    split: str,
) -> None:
    annotation_path = (
        RAW_ROOT
        / dataset_name
        / "coco"
        / split
        / "_annotations.coco.json"
    )

    print()
    print("=" * 80)
    print(dataset_name)
    print("split:", split)
    print("path:", annotation_path)

    if not annotation_path.exists():
        print("MISSING ANNOTATION FILE")
        return

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    images = coco.get(
        "images",
        [],
    )

    annotations = coco.get(
        "annotations",
        [],
    )

    categories = coco.get(
        "categories",
        [],
    )

    category_map = {
        category["id"]: category["name"]
        for category in categories
    }

    annotation_counts = Counter(
        annotation["category_id"]
        for annotation in annotations
    )

    print(
        "images:",
        len(images),
    )

    print(
        "annotations:",
        len(annotations),
    )

    print(
        "categories:",
        categories,
    )

    print(
        "annotation counts:",
    )

    for category_id, count in sorted(
        annotation_counts.items()
    ):
        print(
            f"  id={category_id} "
            f"name={category_map.get(category_id)!r} "
            f"count={count}"
        )


def main() -> None:
    for dataset_name in DATASETS:
        for split in SPLITS:
            inspect_split(
                dataset_name,
                split,
            )


if __name__ == "__main__":
    main()