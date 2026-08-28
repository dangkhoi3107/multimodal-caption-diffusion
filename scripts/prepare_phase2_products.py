import argparse
import json
import shutil
from collections import Counter
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


PREPROCESSING_VERSION = "phase2_multiclass_v1"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_data.yaml"),
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


def load_rgb_image(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    with Image.open(path) as image:
        image = image.convert("RGB")

        return np.asarray(
            image,
            dtype=np.uint8,
        )


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

    Image.fromarray(
        image
    ).save(
        temporary_path
    )

    temporary_path.replace(
        path
    )


def write_json_atomic(
    data,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    temporary_path.replace(
        path
    )


def write_jsonl_atomic(
    records: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    temporary_path.replace(
        path
    )


def validate_sources(
    sources: list[dict],
) -> None:
    if not sources:
        raise ValueError(
            "Config must contain at least one source"
        )

    class_ids = [
        int(source["class_id"])
        for source in sources
    ]

    if len(class_ids) != len(
        set(class_ids)
    ):
        raise ValueError(
            "Duplicate class_id in sources"
        )

    expected_ids = list(
        range(len(class_ids))
    )

    if sorted(class_ids) != expected_ids:
        raise ValueError(
            "class_id must be contiguous: "
            f"{expected_ids}"
        )

    class_names = [
        str(source["class_name"])
        for source in sources
    ]

    if len(class_names) != len(
        set(class_names)
    ):
        raise ValueError(
            "Duplicate class_name in sources"
        )


def validate_source_document(
    document,
    source: dict,
    split: str,
) -> None:
    target_category_id = int(
        source["coco_category_id"]
    )

    expected_category_name = str(
        source["coco_category_name"]
    )

    categories = {
        category.id: category.name
        for category in document.categories
    }

    if target_category_id not in categories:
        raise ValueError(
            f"{source['class_name']} / {split}: "
            f"missing coco_category_id="
            f"{target_category_id}"
        )

    actual_name = categories[
        target_category_id
    ]

    if actual_name != expected_category_name:
        raise ValueError(
            f"{source['class_name']} / {split}: "
            "COCO category name mismatch. "
            f"Expected {expected_category_name!r}, "
            f"got {actual_name!r}"
        )


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    output_root = Path(
        config["output_root"]
    )

    splits = list(
        config["splits"]
    )

    annotation_filename = str(
        config["annotation_filename"]
    )

    image_size = int(
        config["image_size"]
    )

    margin_ratio = float(
        config["margin_ratio"]
    )

    background_mode = str(
        config["background_mode"]
    )

    sources = list(
        config["sources"]
    )

    validate_sources(
        sources
    )

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
    # Save stable project class mapping
    # ---------------------------------

    classes = []

    for source in sorted(
        sources,
        key=lambda item: int(
            item["class_id"]
        ),
    ):
        classes.append(
            {
                "class_id": int(
                    source["class_id"]
                ),
                "class_name": str(
                    source["class_name"]
                ),
                "source_raw_root": str(
                    source["raw_root"]
                ),
                "coco_category_id": int(
                    source[
                        "coco_category_id"
                    ]
                ),
                "coco_category_name": str(
                    source[
                        "coco_category_name"
                    ]
                ),
            }
        )

    write_json_atomic(
        classes,
        output_root / "classes.json",
    )

    # ---------------------------------
    # Summary structure
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
        "num_classes": len(
            sources
        ),
        "splits": {},
    }

    # ---------------------------------
    # Process each split
    # ---------------------------------

    for split in splits:
        print()
        print("=" * 80)
        print(
            f"Processing split: {split}"
        )

        split_output_dir = (
            output_root
            / split
        )

        split_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_records = []

        processed_total = 0
        skipped_total = 0

        class_counts = Counter()
        source_summaries = []

        # ---------------------------------
        # Process every source dataset
        # ---------------------------------

        for source in sorted(
            sources,
            key=lambda item: int(
                item["class_id"]
            ),
        ):
            class_id = int(
                source["class_id"]
            )

            class_name = str(
                source["class_name"]
            )

            raw_root = Path(
                source["raw_root"]
            )

            target_category_id = int(
                source[
                    "coco_category_id"
                ]
            )

            annotation_path = (
                raw_root
                / split
                / annotation_filename
            )

            print()
            print(
                f"class_id={class_id}"
            )
            print(
                f"class_name={class_name}"
            )
            print(
                f"annotation={annotation_path}"
            )

            document = load_coco(
                annotation_path
            )

            validate_source_document(
                document=document,
                source=source,
                split=split,
            )

            indexes = build_coco_indexes(
                document
            )

            source_processed = 0
            source_skipped = 0
            errors = []

            for instance in iter_instances(
                indexes
            ):
                annotation = (
                    instance.annotation
                )

                # Ignore Stock supercategory or
                # any other category not selected
                # for this source.
                if (
                    annotation.category_id
                    != target_category_id
                ):
                    continue

                source_image_path = (
                    raw_root
                    / split
                    / instance.image.file_name
                )

                try:
                    image = load_rgb_image(
                        source_image_path
                    )

                    (
                        actual_height,
                        actual_width,
                    ) = image.shape[:2]

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

                    # IMPORTANT:
                    # COCO IDs can repeat between
                    # source datasets, so prefix
                    # with project class ID.
                    output_filename = (
                        f"class_{class_id:02d}_"
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

                        "class_id": class_id,
                        "class_name": (
                            class_name
                        ),

                        "source_dataset": (
                            class_name
                        ),

                        "source_raw_root": str(
                            raw_root
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
                            annotation.category_id
                        ),

                        "coco_category_name": (
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

                        "image_size": (
                            image_size
                        ),

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

                    source_processed += 1
                    processed_total += 1

                    class_counts[
                        class_id
                    ] += 1

                except Exception as error:
                    source_skipped += 1
                    skipped_total += 1

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
                            "error": str(
                                error
                            ),
                        }
                    )

            source_summaries.append(
                {
                    "class_id": (
                        class_id
                    ),
                    "class_name": (
                        class_name
                    ),
                    "source_images": (
                        len(document.images)
                    ),
                    "source_annotations": (
                        len(
                            document.annotations
                        )
                    ),
                    "processed": (
                        source_processed
                    ),
                    "skipped": (
                        source_skipped
                    ),
                    "errors": errors,
                }
            )

            print(
                f"processed="
                f"{source_processed}, "
                f"skipped="
                f"{source_skipped}"
            )

        # ---------------------------------
        # Stable metadata ordering
        # ---------------------------------

        metadata_records.sort(
            key=lambda item: (
                item["class_id"],
                item["image_id"],
                item["annotation_id"],
            )
        )

        metadata_path = (
            output_root
            / f"{split}.jsonl"
        )

        write_jsonl_atomic(
            metadata_records,
            metadata_path,
        )

        summary[
            "splits"
        ][split] = {
            "processed": (
                processed_total
            ),
            "skipped": (
                skipped_total
            ),
            "class_counts": {
                str(class_id): (
                    class_counts[
                        class_id
                    ]
                )
                for class_id in sorted(
                    class_counts
                )
            },
            "sources": (
                source_summaries
            ),
        }

        print()
        print(
            f"Split total: "
            f"processed={processed_total}, "
            f"skipped={skipped_total}"
        )

        print(
            "Class counts:",
            dict(
                sorted(
                    class_counts.items()
                )
            ),
        )

    # ---------------------------------
    # Save summary
    # ---------------------------------

    write_json_atomic(
        summary,
        output_root
        / "preprocessing_summary.json",
    )

    print()
    print("=" * 80)

    print(
        "Processed multi-class dataset "
        f"saved to: {output_root}"
    )


if __name__ == "__main__":
    main()