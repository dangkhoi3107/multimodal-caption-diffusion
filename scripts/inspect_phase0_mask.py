from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.coco import (
    build_coco_indexes,
    iter_instances,
    load_coco,
)
from src.data.product_preprocessing import (
    polygon_to_mask,
)


def main():
    split_root = Path(
        "data/raw/products/"
        "lifebuoy_handwash_vitamin_protection_400g/"
        "coco/train"
    )

    annotation_path = (
        split_root / "_annotations.coco.json"
    )

    document = load_coco(annotation_path)

    indexes = build_coco_indexes(document)

    instances = list(
        iter_instances(indexes)
    )

    instance = instances[0]

    image_path = (
        split_root / instance.image.file_name
    )

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    mask = polygon_to_mask(
        polygons=instance.annotation.segmentation,
        height=instance.image.height,
        width=instance.image.width,
    )

    masked_image = image_rgb.copy()

    masked_image[
        mask == 0
    ] = 255

    print(
        "image:",
        instance.image.file_name,
    )

    print(
        "image shape:",
        image_rgb.shape,
    )

    print(
        "mask shape:",
        mask.shape,
    )

    print(
        "annotation id:",
        instance.annotation.id,
    )

    print(
        "category:",
        instance.category.name,
    )

    print(
        "mask foreground pixels:",
        int(mask.sum()),
    )

    output_dir = Path(
        "outputs/phase0_data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")

    axes[1].imshow(
        mask,
        cmap="gray",
    )
    axes[1].set_title("Binary Mask")

    axes[2].imshow(masked_image)
    axes[2].set_title(
        "Object on White Background"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        output_dir
        / "debug_real_mask.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"saved: {output_path}"
    )


if __name__ == "__main__":
    main()