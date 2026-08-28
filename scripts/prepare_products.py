import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.data.coco import (
    build_coco_indexes,
    iter_instances,
    load_coco,
)
from src.data.product_preprocessing import (
    prepare_instance,
)


PREPROCESSING_VERSION = "phase0_v1"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase0_data.yaml"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Config root must be a mapping"
        )

    return config


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    with Image.open(path) as image:
        image = image.convert("RGB")

        array = np.asarray(
            image,
            dtype=np.uint8,
        )

    return array


def save_png_atomic(
    image: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        ".tmp.png"
    )

    Image.fromarray(image).save(
        temporary_path
    )

    temporary_path.replace(path)


def write_json_atomic(
    data,
    path: Path,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


def write_jsonl_atomic(
    records: list[dict],
    path: Path,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    temporary_path.replace(path)


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    raw_root = Path(
        config["raw_root"]
    )

    output_root = Path(
        config["output_root"]
    )

    splits = list(
        config["splits"]
    )

    annotation_filename = config[
        "annotation_filename"
    ]

    ignored_category_ids = set(
        config.get(
            "ignored_category_ids",
            [],
        )
    )

    image_size = int(
        config["image_size"]
    )

    margin_ratio = float(
        config["margin_ratio"]
    )

    background_mode = config[
        "background_mode"
    ]

    if background_mode != "white":
        raise ValueError(
            "Only background_mode=white "
            "is currently supported"
        )

    background = (
        255,
        255,
        255,
    )

    # ---------------------------------
    # Prepare output directory
    # ---------------------------------

    if output_root.exists():
        has_content = any(
            output_root.iterdir()
        )

        if has_content:
            if not args.overwrite:
                raise RuntimeError(
                    f"{output_root} already contains data. "
                    "Use --overwrite to regenerate."
                )

            shutil.rmtree(
                output_root
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------
    # Load all COCO documents first
    # ---------------------------------

    documents = {}

    for split in splits:
        annotation_path = (
            raw_root
            / split
            / annotation_filename
        )

        document = load_coco(
            annotation_path
        )

        documents[split] = document

    # ---------------------------------
    # Build class mapping
    # ---------------------------------

    category_names: dict[int, str] = {}

    for document in documents.values():
        for category in document.categories:
            if category.id in ignored_category_ids:
                continue

            existing_name = category_names.get(
                category.id
            )

            if (
                existing_name is not None
                and existing_name != category.name
            ):
                raise ValueError(
                    "Category name mismatch for "
                    f"category_id={category.id}"
                )

            category_names[
                category.id
            ] = category.name

    coco_category_ids = sorted(
        category_names
    )

    class_id_by_coco_category_id = {
        coco_category_id: class_id
        for class_id, coco_category_id
        in enumerate(coco_category_ids)
    }

    classes = []

    for coco_category_id in coco_category_ids:
        classes.append(
            {
                "class_id": (
                    class_id_by_coco_category_id[
                        coco_category_id
                    ]
                ),
                "coco_category_id": (
                    coco_category_id
                ),
                "class_name": (
                    category_names[
                        coco_category_id
                    ]
                ),
            }
        )

    write_json_atomic(
        classes,
        output_root / "classes.json",
    )

    # ---------------------------------
    # Process every split
    # ---------------------------------

    summary = {
        "preprocessing_version": (
            PREPROCESSING_VERSION
        ),
        "image_size": image_size,
        "margin_ratio": margin_ratio,
        "background_mode": (
            background_mode
        ),
        "ignored_category_ids": sorted(
            ignored_category_ids
        ),
        "splits": {},
    }

    for split in splits:
        print()
        print(
            f"Processing split: {split}"
        )

        document = documents[split]

        indexes = build_coco_indexes(
            document
        )

        split_output_dir = (
            output_root / split
        )

        split_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_records = []

        processed = 0
        skipped = 0
        errors = []

        for instance in iter_instances(
            indexes
        ):
            annotation = (
                instance.annotation
            )

            coco_category_id = (
                annotation.category_id
            )

            if (
                coco_category_id
                in ignored_category_ids
            ):
                skipped += 1
                continue

            class_id = (
                class_id_by_coco_category_id[
                    coco_category_id
                ]
            )

            source_image_path = (
                raw_root
                / split
                / instance.image.file_name
            )

            try:
                image = load_rgb_image(
                    source_image_path
                )

                actual_height, actual_width = (
                    image.shape[:2]
                )

                if (
                    actual_width
                    != instance.image.width
                    or actual_height
                    != instance.image.height
                ):
                    raise ValueError(
                        "Image dimensions do not "
                        "match COCO metadata"
                    )

                prepared = prepare_instance(
                    image=image,
                    polygons=(
                        annotation.segmentation
                    ),
                    image_size=image_size,
                    margin_ratio=margin_ratio,
                    background=background,
                )

                output_filename = (
                    f"{split}_"
                    f"img_{instance.image.id:06d}_"
                    f"ann_{annotation.id:06d}.png"
                )

                output_path = (
                    split_output_dir
                    / output_filename
                )

                save_png_atomic(
                    prepared.image,
                    output_path,
                )

                metadata = {
                    "split": split,
                    "file_name": (
                        f"{split}/"
                        f"{output_filename}"
                    ),
                    "source_image": (
                        instance.image.file_name
                    ),
                    "source_name": (
                        instance.image.source_name
                    ),
                    "image_id": (
                        instance.image.id
                    ),
                    "annotation_id": (
                        annotation.id
                    ),
                    "coco_category_id": (
                        coco_category_id
                    ),
                    "class_id": class_id,
                    "class_name": (
                        instance.category.name
                    ),
                    "tight_bounds": list(
                        prepared.tight_bounds
                    ),
                    "expanded_bounds": list(
                        prepared.expanded_bounds
                    ),
                    "foreground_pixels": (
                        prepared.foreground_pixels
                    ),
                    "image_size": image_size,
                    "margin_ratio": (
                        margin_ratio
                    ),
                    "background_mode": (
                        background_mode
                    ),
                    "preprocessing_version": (
                        PREPROCESSING_VERSION
                    ),
                }

                metadata_records.append(
                    metadata
                )

                processed += 1

            except Exception as error:
                skipped += 1

                errors.append(
                    {
                        "image_id": (
                            instance.image.id
                        ),
                        "annotation_id": (
                            annotation.id
                        ),
                        "source_image": (
                            instance.image.file_name
                        ),
                        "error": str(error),
                    }
                )

        metadata_path = (
            output_root
            / f"{split}.jsonl"
        )

        write_jsonl_atomic(
            metadata_records,
            metadata_path,
        )

        summary["splits"][split] = {
            "source_images": (
                len(document.images)
            ),
            "source_annotations": (
                len(document.annotations)
            ),
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }

        print(
            f"processed={processed}, "
            f"skipped={skipped}"
        )

    write_json_atomic(
        summary,
        output_root
        / "preprocessing_summary.json",
    )

    print()
    print(
        f"Processed dataset saved to: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()