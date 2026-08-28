"""Utilities for parsing and indexing COCO-format annotations.

This module only handles COCO metadata:
COCO JSON -> validation -> typed records -> lookup indexes.

It does NOT:
- open or resize image pixels;
- create masks or crops;
- filter categories based on model config;
- map COCO category IDs to model class IDs;
- write processed outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    source_name: str | None = None


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str
    supercategory: str | None = None


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: tuple[tuple[float, ...], ...]
    bbox: tuple[float, float, float, float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CocoDocument:
    images: tuple[CocoImage, ...]
    annotations: tuple[CocoAnnotation, ...]
    categories: tuple[CocoCategory, ...]


@dataclass(frozen=True)
class CocoIndexes:
    images_by_id: dict[int, CocoImage]
    annotations_by_image_id: dict[int, tuple[CocoAnnotation, ...]]
    categories_by_id: dict[int, CocoCategory]

@dataclass(frozen=True)
class ProductInstance:
    image: CocoImage
    annotation: CocoAnnotation
    category: CocoCategory


def _require_field(
    record: dict[str, Any],
    field: str,
    record_type: str,
) -> Any:
    if field not in record:
        raise ValueError(
            f"{record_type} is missing required field: {field}"
        )
    return record[field]


def _require_record(
    record: Any,
    record_type: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(
            f"{record_type} must be a JSON object"
        )
    return record


def _parse_image(record: dict[str, Any]) -> CocoImage:
    image_id = int(
        _require_field(record, "id", "image")
    )
    file_name = str(
        _require_field(record, "file_name", "image")
    )
    width = int(
        _require_field(record, "width", "image")
    )
    height = int(
        _require_field(record, "height", "image")
    )

    if not file_name:
        raise ValueError(
            "image file_name must not be empty"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "image width and height must be positive"
        )

    extra = record.get("extra", {})
    source_name: str | None = None

    if isinstance(extra, dict):
        raw_source_name = extra.get("name")
        if raw_source_name is not None:
            source_name = str(raw_source_name)

    return CocoImage(
        id=image_id,
        file_name=file_name,
        width=width,
        height=height,
        source_name=source_name,
    )


def _parse_category(
    record: dict[str, Any],
) -> CocoCategory:
    category_id = int(
        _require_field(record, "id", "category")
    )
    name = str(
        _require_field(record, "name", "category")
    )

    if not name:
        raise ValueError(
            "category name must not be empty"
        )

    supercategory_raw = record.get("supercategory")
    supercategory = (
        str(supercategory_raw)
        if supercategory_raw is not None
        else None
    )

    return CocoCategory(
        id=category_id,
        name=name,
        supercategory=supercategory,
    )


def _parse_annotation(
    record: dict[str, Any],
) -> CocoAnnotation:
    annotation_id = int(
        _require_field(record, "id", "annotation")
    )
    image_id = int(
        _require_field(
            record,
            "image_id",
            "annotation",
        )
    )
    category_id = int(
        _require_field(
            record,
            "category_id",
            "annotation",
        )
    )

    segmentation_raw = _require_field(
        record,
        "segmentation",
        "annotation",
    )

    if isinstance(segmentation_raw, dict):
        raise ValueError(
            "RLE segmentation is not supported"
        )

    if not isinstance(segmentation_raw, list):
        raise ValueError(
            "annotation segmentation must be a list"
        )

    if not segmentation_raw:
        raise ValueError(
            "annotation segmentation must contain at least one polygon"
        )

    polygons: list[tuple[float, ...]] = []

    for polygon_index, polygon in enumerate(
        segmentation_raw
    ):
        if not isinstance(polygon, list):
            raise ValueError(
                "annotation segmentation polygon "
                "must be a list"
            )

        if len(polygon) < 6 or len(polygon) % 2 != 0:
            raise ValueError(
                "polygon must contain an even number "
                "of coordinates and at least 6 values"
            )

        try:
            coordinates = tuple(
                float(value)
                for value in polygon
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "polygon coordinates must be numeric "
                f"(polygon index {polygon_index})"
            ) from exc

        if not all(
            math.isfinite(value)
            for value in coordinates
        ):
            raise ValueError(
                "polygon coordinates must be finite"
            )

        polygons.append(coordinates)

    bbox_raw = _require_field(
        record,
        "bbox",
        "annotation",
    )

    if (
        not isinstance(bbox_raw, list)
        or len(bbox_raw) != 4
    ):
        raise ValueError(
            "annotation bbox must contain exactly 4 values"
        )

    try:
        bbox_values = tuple(
            float(value)
            for value in bbox_raw
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "annotation bbox values must be numeric"
        ) from exc

    if not all(
        math.isfinite(value)
        for value in bbox_values
    ):
        raise ValueError(
            "annotation bbox values must be finite"
        )

    x, y, width, height = bbox_values

    if width <= 0 or height <= 0:
        raise ValueError(
            "annotation bbox width and height "
            "must be positive"
        )

    bbox: tuple[float, float, float, float] = (
        x,
        y,
        width,
        height,
    )

    try:
        area = float(
            _require_field(
                record,
                "area",
                "annotation",
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "annotation area must be numeric"
        ) from exc

    if not math.isfinite(area) or area <= 0:
        raise ValueError(
            "annotation area must be positive and finite"
        )

    try:
        iscrowd = int(record.get("iscrowd", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "annotation iscrowd must be an integer"
        ) from exc

    if iscrowd not in (0, 1):
        raise ValueError(
            "annotation iscrowd must be 0 or 1"
        )

    return CocoAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=category_id,
        segmentation=tuple(polygons),
        bbox=bbox,
        area=area,
        iscrowd=iscrowd,
    )


def _validate_unique_ids(
    values: tuple[Any, ...],
    name: str,
) -> None:
    ids = [value.id for value in values]

    if len(ids) == len(set(ids)):
        return

    seen: set[int] = set()
    duplicates: set[int] = set()

    for value_id in ids:
        if value_id in seen:
            duplicates.add(value_id)
        seen.add(value_id)

    duplicate_text = ", ".join(
        str(value_id)
        for value_id in sorted(duplicates)
    )

    raise ValueError(
        f"duplicate {name} id detected: {duplicate_text}"
    )


def load_coco(path: Path) -> CocoDocument:
    """Load and validate one COCO annotation JSON document.

    IDs are validated only inside this document. This is intentional because
    train/valid/test exports may reuse image and annotation IDs.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"COCO annotation path is not a file: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid COCO JSON: {path}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(
            "COCO root must be a JSON object"
        )

    images_raw = _require_field(
        raw,
        "images",
        "COCO document",
    )
    annotations_raw = _require_field(
        raw,
        "annotations",
        "COCO document",
    )
    categories_raw = _require_field(
        raw,
        "categories",
        "COCO document",
    )

    if not isinstance(images_raw, list):
        raise ValueError(
            "COCO images must be a list"
        )

    if not isinstance(annotations_raw, list):
        raise ValueError(
            "COCO annotations must be a list"
        )

    if not isinstance(categories_raw, list):
        raise ValueError(
            "COCO categories must be a list"
        )

    images = tuple(
        _parse_image(
            _require_record(record, "image")
        )
        for record in images_raw
    )

    annotations = tuple(
        _parse_annotation(
            _require_record(record, "annotation")
        )
        for record in annotations_raw
    )

    categories = tuple(
        _parse_category(
            _require_record(record, "category")
        )
        for record in categories_raw
    )

    _validate_unique_ids(images, "image")
    _validate_unique_ids(
        annotations,
        "annotation",
    )
    _validate_unique_ids(
        categories,
        "category",
    )

    image_ids = {
        image.id
        for image in images
    }
    category_ids = {
        category.id
        for category in categories
    }

    for annotation in annotations:
        if annotation.image_id not in image_ids:
            raise ValueError(
                "annotation references missing "
                f"image_id={annotation.image_id}"
            )

        if annotation.category_id not in category_ids:
            raise ValueError(
                "annotation references missing "
                f"category_id={annotation.category_id}"
            )

    return CocoDocument(
        images=images,
        annotations=annotations,
        categories=categories,
    )


def build_coco_indexes(
    document: CocoDocument,
) -> CocoIndexes:
    """Build lookup maps for one parsed COCO document."""
    images_by_id = {
        image.id: image
        for image in document.images
    }

    categories_by_id = {
        category.id: category
        for category in document.categories
    }

    annotation_lists: dict[
        int,
        list[CocoAnnotation],
    ] = {
        image.id: []
        for image in document.images
    }

    for annotation in document.annotations:
        annotation_lists[
            annotation.image_id
        ].append(annotation)

    annotations_by_image_id = {
        image_id: tuple(annotations)
        for image_id, annotations
        in annotation_lists.items()
    }

    return CocoIndexes(
        images_by_id=images_by_id,
        annotations_by_image_id=annotations_by_image_id,
        categories_by_id=categories_by_id,
    )




def iter_instances(
    indexes: CocoIndexes,
):
    for image_id, annotations in (
        indexes.annotations_by_image_id.items()
    ):
        image = indexes.images_by_id[image_id]

        for annotation in annotations:
            category = indexes.categories_by_id[
                annotation.category_id
            ]

            yield ProductInstance(
                image=image,
                annotation=annotation,
                category=category,
            )