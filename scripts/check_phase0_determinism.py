import hashlib
import subprocess
import sys
from pathlib import Path


PROCESSED_ROOT = Path(
    "data/processed/products_64"
)

METADATA_FILES = (
    "train.jsonl",
    "valid.jsonl",
    "test.jsonl",
    "classes.json",
    "preprocessing_summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def metadata_hashes() -> dict[str, str]:
    return {
        name: sha256_file(
            PROCESSED_ROOT / name
        )
        for name in METADATA_FILES
    }


def image_hashes() -> dict[str, str]:
    hashes = {}

    for split in (
        "train",
        "valid",
        "test",
    ):
        split_root = (
            PROCESSED_ROOT / split
        )

        for path in sorted(
            split_root.glob("*.png")
        ):
            relative_path = str(
                path.relative_to(
                    PROCESSED_ROOT
                )
            )

            hashes[relative_path] = (
                sha256_file(path)
            )

    return hashes


def main():
    print(
        "Computing hashes before regeneration..."
    )

    metadata_before = (
        metadata_hashes()
    )

    images_before = (
        image_hashes()
    )

    print(
        f"Images before: "
        f"{len(images_before)}"
    )

    print()
    print(
        "Regenerating processed dataset..."
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_products",
            "--config",
            "configs/phase0_data.yaml",
            "--overwrite",
        ],
        check=True,
    )

    print()
    print(
        "Computing hashes after regeneration..."
    )

    metadata_after = (
        metadata_hashes()
    )

    images_after = (
        image_hashes()
    )

    metadata_match = (
        metadata_before
        == metadata_after
    )

    images_match = (
        images_before
        == images_after
    )

    print()
    print(
        "Metadata deterministic:",
        metadata_match,
    )

    print(
        "Images deterministic:",
        images_match,
    )

    print(
        "Image count:",
        len(images_after),
    )

    if not metadata_match:
        print()
        print(
            "Metadata differences:"
        )

        for name in METADATA_FILES:
            if (
                metadata_before[name]
                != metadata_after[name]
            ):
                print(
                    f"- {name}"
                )

    if not images_match:
        print()
        print(
            "Image differences:"
        )

        all_paths = (
            set(images_before)
            | set(images_after)
        )

        for path in sorted(
            all_paths
        ):
            if (
                images_before.get(path)
                != images_after.get(path)
            ):
                print(
                    f"- {path}"
                )

    if not (
        metadata_match
        and images_match
    ):
        raise RuntimeError(
            "Phase 0 preprocessing "
            "is not deterministic"
        )

    print()
    print(
        "PASS: Phase 0 regeneration "
        "is deterministic."
    )


if __name__ == "__main__":
    main()