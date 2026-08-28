import numpy as np

from src.data.product_preprocessing import (
    apply_mask,
    crop_to_bounds,
    expand_bounds,
    letterbox_square,
    mask_bounds,
    polygon_to_mask,
    prepare_instance,
)
import pytest


def test_polygon_to_mask_rectangle():
    polygons = (
        (
            1.0, 1.0,
            3.0, 1.0,
            3.0, 3.0,
            1.0, 3.0,
        ),
    )

    mask = polygon_to_mask(
        polygons=polygons,
        height=5,
        width=5,
    )

    assert mask.shape == (5, 5)
    assert mask.dtype == np.uint8

    assert set(
        np.unique(mask).tolist()
    ).issubset({0, 1})

    assert mask.sum() == 9


def test_polygon_to_mask_background_is_zero():
    polygons = (
        (
            1.0, 1.0,
            3.0, 1.0,
            3.0, 3.0,
            1.0, 3.0,
        ),
    )

    mask = polygon_to_mask(
        polygons=polygons,
        height=5,
        width=5,
    )

    assert mask[0, 0] == 0
    assert mask[4, 4] == 0

    assert mask[2, 2] == 1


def test_mask_bounds():
    mask = np.zeros(
        (6, 8),
        dtype=np.uint8,
    )

    mask[1:5, 2:7] = 1

    bounds = mask_bounds(mask)

    assert bounds == (
        2,
        1,
        7,
        5,
    )


def test_mask_bounds_touching_image_edge():
    mask = np.zeros(
        (5, 5),
        dtype=np.uint8,
    )

    mask[0:3, 0:2] = 1

    bounds = mask_bounds(mask)

    assert bounds == (
        0,
        0,
        2,
        3,
    )

def test_mask_bounds_empty_mask_raises_error():
    mask = np.zeros(
        (5, 5),
        dtype=np.uint8,
    )

    try:
        mask_bounds(mask)

        assert False

    except ValueError as error:
        assert (
            "no foreground pixels"
            in str(error)
        )



def test_mask_bounds_empty_mask_raises_error():
    mask = np.zeros(
        (5, 5),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="no foreground pixels",
    ):
        mask_bounds(mask)


def test_expand_bounds():
    bounds = (
        100,
        50,
        200,
        250,
    )

    expanded = expand_bounds(
        bounds=bounds,
        margin_ratio=0.10,
        image_height=400,
        image_width=400,
    )

    assert expanded == (
        90,
        30,
        210,
        270,
    )

def test_expand_bounds_clips_to_image_edges():
    bounds = (
        0,
        5,
        40,
        100,
    )

    expanded = expand_bounds(
        bounds=bounds,
        margin_ratio=0.10,
        image_height=200,
        image_width=300,
    )

    assert expanded[0] == 0
    assert expanded[1] >= 0
    assert expanded[2] <= 300
    assert expanded[3] <= 200


def test_expand_bounds_clips_right_and_bottom_edges():
    bounds = (
        250,
        150,
        300,
        200,
    )

    expanded = expand_bounds(
        bounds=bounds,
        margin_ratio=0.20,
        image_height=200,
        image_width=300,
    )

    assert expanded[2] == 300
    assert expanded[3] == 200

def test_expand_bounds_negative_margin_raises_error():
    import pytest

    with pytest.raises(
        ValueError,
        match="margin_ratio",
    ):
        expand_bounds(
            bounds=(10, 10, 20, 20),
            margin_ratio=-0.1,
            image_height=100,
            image_width=100,
        )


def test_apply_mask_white_background():
    image = np.array(
        [
            [
                [10, 20, 30],
                [40, 50, 60],
            ],
            [
                [70, 80, 90],
                [100, 110, 120],
            ],
        ],
        dtype=np.uint8,
    )

    mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    output = apply_mask(
        image=image,
        mask=mask,
        background=(255, 255, 255),
    )

    assert np.array_equal(
        output[0, 0],
        [10, 20, 30],
    )

    assert np.array_equal(
        output[1, 1],
        [100, 110, 120],
    )

    assert np.array_equal(
        output[0, 1],
        [255, 255, 255],
    )

    assert np.array_equal(
        output[1, 0],
        [255, 255, 255],
    )

def test_apply_mask_does_not_modify_input():
    image = np.full(
        (4, 4, 3),
        100,
        dtype=np.uint8,
    )

    original = image.copy()

    mask = np.zeros(
        (4, 4),
        dtype=np.uint8,
    )

    apply_mask(
        image=image,
        mask=mask,
    )

    assert np.array_equal(
        image,
        original,
    )


def test_apply_mask_shape_mismatch_raises_error():
    import pytest

    image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    mask = np.zeros(
        (8, 10),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="spatial dimensions",
    ):
        apply_mask(
            image=image,
            mask=mask,
        )

def test_crop_to_bounds():
    image = np.zeros(
        (10, 20, 3),
        dtype=np.uint8,
    )

    bounds = (
        5,
        2,
        15,
        8,
    )

    crop = crop_to_bounds(
        image=image,
        bounds=bounds,
    )

    assert crop.shape == (
        6,
        10,
        3,
    )

    assert crop.dtype == np.uint8



def test_crop_to_bounds_invalid_bounds():
    import pytest

    image = np.zeros(
        (10, 20, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="bounds",
    ):
        crop_to_bounds(
            image=image,
            bounds=(-1, 0, 10, 10),
        )

def test_letterbox_square_output_shape():
    image = np.full(
        (100, 50, 3),
        100,
        dtype=np.uint8,
    )

    output = letterbox_square(
        image=image,
        size=64,
    )

    assert output.shape == (
        64,
        64,
        3,
    )

    assert output.dtype == np.uint8


def test_letterbox_square_preserves_portrait_ratio():
    image = np.full(
        (100, 50, 3),
        100,
        dtype=np.uint8,
    )

    output = letterbox_square(
        image=image,
        size=64,
        fill=(255, 255, 255),
    )

    non_background = np.any(
        output != 255,
        axis=2,
    )

    ys, xs = np.nonzero(
        non_background
    )

    content_height = (
        ys.max() - ys.min() + 1
    )

    content_width = (
        xs.max() - xs.min() + 1
    )

    assert content_height == 64
    assert content_width == 32

def test_letterbox_square_preserves_landscape_ratio():
    image = np.full(
        (40, 80, 3),
        100,
        dtype=np.uint8,
    )

    output = letterbox_square(
        image=image,
        size=64,
        fill=(255, 255, 255),
    )

    non_background = np.any(
        output != 255,
        axis=2,
    )

    ys, xs = np.nonzero(
        non_background
    )

    content_height = (
        ys.max() - ys.min() + 1
    )

    content_width = (
        xs.max() - xs.min() + 1
    )

    assert content_height == 32
    assert content_width == 64


def test_prepare_instance_output():
    image = np.full(
        (100, 100, 3),
        50,
        dtype=np.uint8,
    )

    polygons = (
        (
            30.0, 20.0,
            70.0, 20.0,
            70.0, 80.0,
            30.0, 80.0,
        ),
    )

    result = prepare_instance(
        image=image,
        polygons=polygons,
        image_size=64,
        margin_ratio=0.10,
    )

    assert result.image.shape == (
        64,
        64,
        3,
    )

    assert result.image.dtype == np.uint8

    assert result.foreground_pixels > 0

    assert (
        result.expanded_bounds[0]
        <= result.tight_bounds[0]
    )

    assert (
        result.expanded_bounds[1]
        <= result.tight_bounds[1]
    )

    assert (
        result.expanded_bounds[2]
        >= result.tight_bounds[2]
    )

    assert (
        result.expanded_bounds[3]
        >= result.tight_bounds[3]
    )