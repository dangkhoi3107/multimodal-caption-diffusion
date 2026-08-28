from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.data.coco import (
    build_coco_indexes,
    iter_instances,
    load_coco,
)
from src.data.product_preprocessing import (
    prepare_instance,
)


def load_rgb(path: Path):
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def touches_edge(
    bounds: tuple[int, int, int, int],
    height: int,
    width: int,
) -> bool:
    left, top, right, bottom = bounds

    return (
        left == 0
        or top == 0
        or right == width
        or bottom == height
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

    normal_example = None
    edge_example = None

    for instance in instances:
        image_path = (
            split_root
            / instance.image.file_name
        )

        image_rgb = load_rgb(image_path)

        prepared = prepare_instance(
            image=image_rgb,
            polygons=instance.annotation.segmentation,
            image_size=64,
            margin_ratio=0.10,
        )

        is_edge = touches_edge(
            prepared.tight_bounds,
            height=instance.image.height,
            width=instance.image.width,
        )

        if is_edge and edge_example is None:
            edge_example = (
                instance,
                image_rgb,
                prepared,
            )

        if not is_edge and normal_example is None:
            normal_example = (
                instance,
                image_rgb,
                prepared,
            )

        if (
            normal_example is not None
            and edge_example is not None
        ):
            break

    if normal_example is None:
        raise RuntimeError(
            "Could not find normal instance"
        )

    if edge_example is None:
        raise RuntimeError(
            "Could not find edge-touching instance"
        )

    examples = [
        ("Normal", normal_example),
        ("Touches edge", edge_example),
    ]

    output_dir = Path(
        "outputs/phase0_data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
    )

    for row, (
        label,
        example,
    ) in enumerate(examples):
        instance, original, prepared = example

        axes[row, 0].imshow(original)

        axes[row, 0].set_title(
            f"{label} - Original"
        )

        axes[row, 1].imshow(
            prepared.image
        )

        axes[row, 1].set_title(
            f"{label} - 64x64 crop"
        )

        print()
        print(label)
        print(
            "file:",
            instance.image.file_name,
        )
        print(
            "annotation id:",
            instance.annotation.id,
        )
        print(
            "original shape:",
            original.shape,
        )
        print(
            "tight bounds:",
            prepared.tight_bounds,
        )
        print(
            "expanded bounds:",
            prepared.expanded_bounds,
        )
        print(
            "foreground pixels:",
            prepared.foreground_pixels,
        )
        print(
            "output shape:",
            prepared.image.shape,
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        output_dir
        / "prepared_crop_examples.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"saved: {output_path}"
    )


if __name__ == "__main__":
    main()