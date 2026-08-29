from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATA_ROOT = Path(
    "data/processed/products_multiclass_64"
)

OUTPUT_ROOT = Path(
    "outputs/phase3_text_conditional"
)

SPLITS = [
    "train",
    "valid",
    "test",
]


CAPTION_SCHEMA = {
    0: {
        "class_name": (
            "dove_body_serum_glow_recharge_547ml"
        ),
        "brand": "dove",
        "product": "body serum",
        "color": "white",
        "package": "bottle",
        "templates": [
            "a white dove body serum bottle",
            "a dove body serum in a white bottle",
            "a white bottle of dove body serum",
        ],
    },
    1: {
        "class_name": (
            "dove_deodorant_niacinamide_omega_40ml"
        ),
        "brand": "dove",
        "product": "deodorant",
        "color": "blue",
        "package": "tube",
        "templates": [
            "a blue dove deodorant tube",
            "a dove deodorant in a blue tube",
            "a blue tube of dove deodorant",
        ],
    },
    2: {
        "class_name": (
            "lifebuoy_handwash_vitamin_protection_400g"
        ),
        "brand": "lifebuoy",
        "product": "handwash",
        "color": "red",
        "package": "pouch",
        "templates": [
            "a red lifebuoy handwash pouch",
            "a lifebuoy handwash in a red pouch",
            "a red pouch of lifebuoy handwash",
        ],
    },
}


def read_jsonl(
    path: Path,
) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            if "class_id" not in record:
                raise ValueError(
                    f"Missing class_id at "
                    f"{path}:{line_number}"
                )

            records.append(
                record
            )

    return records


def choose_template_index(
    record: dict,
    num_templates: int,
) -> int:
    """Deterministic caption choice.

    Prefer annotation_id because it is
    stable across repeated runs.
    """

    annotation_id = int(
        record.get(
            "annotation_id",
            0,
        )
    )

    image_id = int(
        record.get(
            "image_id",
            0,
        )
    )

    return (
        annotation_id
        + image_id
    ) % num_templates


def add_caption(
    record: dict,
) -> dict:
    class_id = int(
        record["class_id"]
    )

    if class_id not in CAPTION_SCHEMA:
        raise ValueError(
            f"Unknown class_id: "
            f"{class_id}"
        )

    schema = CAPTION_SCHEMA[
        class_id
    ]

    expected_name = (
        schema["class_name"]
    )

    actual_name = str(
        record["class_name"]
    )

    if actual_name != expected_name:
        raise ValueError(
            "Class mapping mismatch: "
            f"id={class_id}, "
            f"expected={expected_name}, "
            f"got={actual_name}"
        )

    templates = schema[
        "templates"
    ]

    template_index = (
        choose_template_index(
            record=record,
            num_templates=len(
                templates
            ),
        )
    )

    caption = templates[
        template_index
    ]

    result = dict(
        record
    )

    result[
        "caption"
    ] = caption

    result[
        "caption_template_id"
    ] = template_index

    result[
        "caption_source"
    ] = "phase3_controlled_v1"

    result[
        "text_attributes"
    ] = {
        "brand": schema[
            "brand"
        ],
        "product": schema[
            "product"
        ],
        "color": schema[
            "color"
        ],
        "package": schema[
            "package"
        ],
    }

    return result


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def simple_tokens(
    caption: str,
) -> list[str]:
    """Temporary analysis only.

    This is NOT the Phase 3 tokenizer.
    """

    return (
        caption
        .lower()
        .strip()
        .split()
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "caption_schema_version": (
            "phase3_controlled_v1"
        ),
        "splits": {},
    }

    for split in SPLITS:
        input_path = (
            DATA_ROOT
            / f"{split}.jsonl"
        )

        output_path = (
            DATA_ROOT
            / f"{split}_phase3.jsonl"
        )

        records = read_jsonl(
            input_path
        )

        captioned = [
            add_caption(
                record
            )
            for record in records
        ]

        write_jsonl(
            output_path,
            captioned,
        )

        class_counts = Counter(
            int(
                record["class_id"]
            )
            for record in captioned
        )

        caption_counts = Counter(
            record["caption"]
            for record in captioned
        )

        token_lengths = [
            len(
                simple_tokens(
                    record["caption"]
                )
            )
            for record in captioned
        ]

        vocabulary = sorted(
            {
                token
                for record
                in captioned
                for token
                in simple_tokens(
                    record["caption"]
                )
            }
        )

        report[
            "splits"
        ][
            split
        ] = {
            "num_records": len(
                captioned
            ),
            "class_counts": {
                str(key): value
                for key, value
                in sorted(
                    class_counts.items()
                )
            },
            "num_unique_captions": len(
                caption_counts
            ),
            "min_tokens": min(
                token_lengths
            ),
            "max_tokens": max(
                token_lengths
            ),
            "mean_tokens": (
                sum(
                    token_lengths
                )
                / len(
                    token_lengths
                )
            ),
            "temporary_vocabulary": (
                vocabulary
            ),
        }

        print()
        print(
            f"{split}:"
        )

        print(
            "  records:",
            len(
                captioned
            ),
        )

        print(
            "  classes:",
            dict(
                sorted(
                    class_counts.items()
                )
            ),
        )

        print(
            "  unique captions:",
            len(
                caption_counts
            ),
        )

        print(
            "  token length:",
            f"{min(token_lengths)}"
            f".."
            f"{max(token_lengths)}",
        )

        print(
            "  output:",
            output_path,
        )

        print(
            "  examples:"
        )

        shown = set()

        for record in captioned:
            class_id = int(
                record["class_id"]
            )

            if class_id in shown:
                continue

            shown.add(
                class_id
            )

            print(
                f"    class "
                f"{class_id}: "
                f"{record['caption']}"
            )

    report_path = (
        OUTPUT_ROOT
        / "caption_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=" * 70
    )

    print(
        "Saved report:",
        report_path,
    )


if __name__ == "__main__":
    main()