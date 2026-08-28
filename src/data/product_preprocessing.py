"""
Transfer polygon to mask
"""
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class PreparedCrop:
    image: np.ndarray
    tight_bounds: tuple[int, int, int, int]
    expanded_bounds: tuple[int, int, int, int]
    foreground_pixels: int


def polygon_to_mask(
    polygons: tuple[tuple[float, ...], ...],
    height: int,
    width: int,
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError(
            "height and width must be positive"
        )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for polygon in polygons:
        if len(polygon) < 6:
            raise ValueError(
                "polygon must contain at least "
                "3 points"
            )

        if len(polygon) % 2 != 0:
            raise ValueError(
                "polygon must contain an even "
                "number of coordinates"
            )

        points = np.asarray(
            polygon,
            dtype=np.float32,
        ).reshape(-1, 2)

        if not np.isfinite(points).all():
            raise ValueError(
                "polygon coordinates must be finite"
            )

        points = np.rint(
            points
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [points],
            color=1,
        )

    return mask

def mask_bounds(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape [H, W]"
        )

    foreground_y, foreground_x = np.nonzero(
        mask
    )

    if foreground_x.size == 0:
        raise ValueError(
            "mask contains no foreground pixels"
        )

    left = int(foreground_x.min())
    top = int(foreground_y.min())

    right_exclusive = int(
        foreground_x.max()
    ) + 1

    bottom_exclusive = int(
        foreground_y.max()
    ) + 1

    return (
        left,
        top,
        right_exclusive,
        bottom_exclusive,
    )


def expand_bounds(
    bounds: tuple[int, int, int, int],
    margin_ratio: float,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    if margin_ratio < 0:
        raise ValueError(
            "margin_ratio must be non-negative"
        )

    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    left, top, right, bottom = bounds

    if not (
        0 <= left < right <= image_width
        and 0 <= top < bottom <= image_height
    ):
        raise ValueError(
            "bounds must be inside image dimensions"
        )

    box_width = right - left
    box_height = bottom - top

    margin_x = int(
        round(box_width * margin_ratio)
    )

    margin_y = int(
        round(box_height * margin_ratio)
    )

    expanded_left = max(
        0,
        left - margin_x,
    )

    expanded_top = max(
        0,
        top - margin_y,
    )

    expanded_right = min(
        image_width,
        right + margin_x,
    )

    expanded_bottom = min(
        image_height,
        bottom + margin_y,
    )

    return (
        expanded_left,
        expanded_top,
        expanded_right,
        expanded_bottom,
    )

def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]"
        )

    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape [H, W]"
        )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            "image and mask spatial dimensions must match"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image must have dtype uint8"
        )

    output = image.copy()

    background_array = np.asarray(
        background,
        dtype=np.uint8,
    )

    output[mask == 0] = background_array

    return output


def crop_to_bounds(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]"
        )

    image_height, image_width = image.shape[:2]

    left, top, right, bottom = bounds

    if not (
        0 <= left < right <= image_width
        and 0 <= top < bottom <= image_height
    ):
        raise ValueError(
            "bounds must be inside image dimensions"
        )

    crop = image[
        top:bottom,
        left:right,
    ]

    return crop.copy()

def letterbox_square(
    image: np.ndarray,
    size: int,
    fill: tuple[int, int, int] = (
        255,
        255,
        255,
    ),
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image must have dtype uint8"
        )

    if size <= 0:
        raise ValueError(
            "size must be positive"
        )

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    scale = min(
        size / width,
        size / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=interpolation,
    )

    pad_width = size - new_width
    pad_height = size - new_height

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top

    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill,
    )

    return output


def prepare_instance(
    image: np.ndarray,
    polygons: tuple[tuple[float, ...], ...],
    image_size: int = 64,
    margin_ratio: float = 0.10,
    background: tuple[int, int, int] = (
        255,
        255,
        255,
    ),
) -> PreparedCrop:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image must have dtype uint8"
        )

    height, width = image.shape[:2]

    mask = polygon_to_mask(
        polygons=polygons,
        height=height,
        width=width,
    )

    foreground_pixels = int(
        mask.sum()
    )

    tight_bounds = mask_bounds(
        mask
    )

    expanded_bounds = expand_bounds(
        bounds=tight_bounds,
        margin_ratio=margin_ratio,
        image_height=height,
        image_width=width,
    )

    masked_image = apply_mask(
        image=image,
        mask=mask,
        background=background,
    )

    cropped = crop_to_bounds(
        image=masked_image,
        bounds=expanded_bounds,
    )

    output = letterbox_square(
        image=cropped,
        size=image_size,
        fill=background,
    )

    return PreparedCrop(
        image=output,
        tight_bounds=tight_bounds,
        expanded_bounds=expanded_bounds,
        foreground_pixels=foreground_pixels,
    )