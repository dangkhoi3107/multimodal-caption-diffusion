import json

import pytest

from src.data.coco import (
    build_coco_indexes,
    iter_instances,
    load_coco,
)


def _valid_coco_document():
    return {
        "images": [
            {
                "id": 0,
                "file_name": "image.jpg",
                "width": 640,
                "height": 480,
                "extra": {
                    "name": "source_aug1.jpg",
                },
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "Stock",
                "supercategory": "none",
            },
            {
                "id": 1,
                "name": (
                    "LIFEBUOY_"
                    "NuocRuaTayVitaminBaoVeVuotTroi_400g_tui"
                ),
                "supercategory": "Stock",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "segmentation": [
                    [
                        10.0,
                        20.0,
                        110.0,
                        20.0,
                        110.0,
                        220.0,
                        10.0,
                        220.0,
                    ]
                ],
                "bbox": [
                    10.0,
                    20.0,
                    100.0,
                    200.0,
                ],
                "area": 20000.0,
                "iscrowd": 0,
            }
        ],
    }


def _write_json(path, data):
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def test_load_valid_coco_document(tmp_path):
    path = tmp_path / "_annotations.coco.json"

    _write_json(
        path,
        _valid_coco_document(),
    )

    document = load_coco(path)

    assert len(document.images) == 1
    assert len(document.annotations) == 1
    assert len(document.categories) == 2

    image = document.images[0]

    assert image.id == 0
    assert image.file_name == "image.jpg"
    assert image.width == 640
    assert image.height == 480
    assert image.source_name == "source_aug1.jpg"

    annotation = document.annotations[0]

    assert annotation.id == 1
    assert annotation.image_id == 0
    assert annotation.category_id == 1
    assert annotation.iscrowd == 0

    assert annotation.bbox == (
        10.0,
        20.0,
        100.0,
        200.0,
    )

    assert len(annotation.segmentation) == 1

    assert document.categories[0].id == 0
    assert document.categories[0].name == "Stock"

    assert document.categories[1].id == 1


def test_invalid_polygon_coordinates_raise_error(
    tmp_path,
):
    path = tmp_path / "_annotations.coco.json"

    data = _valid_coco_document()

    data["annotations"][0]["segmentation"] = [
        [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
        ]
    ]

    _write_json(path, data)

    with pytest.raises(
        ValueError,
        match="even number",
    ):
        load_coco(path)


def test_duplicate_image_id_raises_error(
    tmp_path,
):
    path = tmp_path / "_annotations.coco.json"

    data = _valid_coco_document()

    duplicate_image = {
        "id": 0,
        "file_name": "another.jpg",
        "width": 320,
        "height": 240,
    }

    data["images"].append(
        duplicate_image
    )

    _write_json(path, data)

    with pytest.raises(
        ValueError,
        match="duplicate image id",
    ):
        load_coco(path)


def test_missing_image_reference_raises_error(
    tmp_path,
):
    path = tmp_path / "_annotations.coco.json"

    data = _valid_coco_document()

    data["annotations"][0][
        "image_id"
    ] = 999

    _write_json(path, data)

    with pytest.raises(
        ValueError,
        match="missing image_id=999",
    ):
        load_coco(path)


def test_build_coco_indexes(tmp_path):
    path = tmp_path / "_annotations.coco.json"

    _write_json(
        path,
        _valid_coco_document(),
    )

    document = load_coco(path)

    indexes = build_coco_indexes(
        document
    )

    assert set(
        indexes.images_by_id.keys()
    ) == {0}

    assert set(
        indexes.categories_by_id.keys()
    ) == {0, 1}

    assert set(
        indexes.annotations_by_image_id.keys()
    ) == {0}

    image = indexes.images_by_id[0]

    assert image.file_name == "image.jpg"

    category = indexes.categories_by_id[1]

    assert category.name.startswith(
        "LIFEBUOY_"
    )

    annotations = (
        indexes.annotations_by_image_id[0]
    )

    assert len(annotations) == 1
    assert annotations[0].id == 1


def test_index_includes_image_without_annotations(
    tmp_path,
):
    path = tmp_path / "_annotations.coco.json"

    data = _valid_coco_document()

    data["images"].append(
        {
            "id": 1,
            "file_name": "empty.jpg",
            "width": 320,
            "height": 240,
        }
    )

    _write_json(path, data)

    document = load_coco(path)

    indexes = build_coco_indexes(
        document
    )

    assert 1 in indexes.images_by_id

    assert (
        indexes.annotations_by_image_id[1]
        == ()
    )

def test_iter_instances(tmp_path):
    path = tmp_path / "_annotations.coco.json"

    _write_json(
        path,
        _valid_coco_document(),
    )

    document = load_coco(path)

    indexes = build_coco_indexes(
        document
    )

    instances = list(
        iter_instances(indexes)
    )

    assert len(instances) == 1

    instance = instances[0]

    assert instance.image.id == 0
    assert instance.annotation.id == 1
    assert instance.category.id == 1

    assert (
        instance.annotation.image_id
        == instance.image.id
    )

    assert (
        instance.annotation.category_id
        == instance.category.id
    )

def test_iter_instances_multiple_annotations(
    tmp_path,
):
    path = tmp_path / "_annotations.coco.json"

    data = _valid_coco_document()

    second_annotation = {
        "id": 2,
        "image_id": 0,
        "category_id": 1,
        "segmentation": [
            [
                200.0,
                100.0,
                300.0,
                100.0,
                300.0,
                300.0,
                200.0,
                300.0,
            ]
        ],
        "bbox": [
            200.0,
            100.0,
            100.0,
            200.0,
        ],
        "area": 20000.0,
        "iscrowd": 0,
    }

    data["annotations"].append(
        second_annotation
    )

    _write_json(path, data)

    document = load_coco(path)

    indexes = build_coco_indexes(
        document
    )

    instances = list(
        iter_instances(indexes)
    )

    assert len(instances) == 2

    annotation_ids = {
        instance.annotation.id
        for instance in instances
    }

    assert annotation_ids == {1, 2}