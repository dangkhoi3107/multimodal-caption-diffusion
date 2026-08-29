from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from src.text.tokenizer import (
    tokenize,
)
from src.text.vocabulary import (
    build_vocabulary,
)


CONFIG_PATH = Path(
    "configs/phase3_text_conditional.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/phase3_text_conditional"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def read_captions(
    path: Path,
) -> list[str]:
    captions = []

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

            if "caption" not in record:
                raise ValueError(
                    f"Missing caption at "
                    f"{path}:{line_number}"
                )

            caption = str(
                record["caption"]
            ).strip()

            if not caption:
                raise ValueError(
                    f"Empty caption at "
                    f"{path}:{line_number}"
                )

            captions.append(
                caption
            )

    if not captions:
        raise ValueError(
            f"No captions found in {path}"
        )

    return captions


def analyze_split(
    name: str,
    captions: list[str],
    vocabulary,
    max_length: int,
) -> dict:
    token_sequences = [
        tokenize(
            caption
        )
        for caption in captions
    ]

    lengths = [
        len(sequence)
        for sequence in token_sequences
    ]

    unknown_counter = Counter()

    for sequence in token_sequences:
        for token in sequence:
            if token not in (
                vocabulary.token_to_id
            ):
                unknown_counter[
                    token
                ] += 1

    too_long_count = sum(
        length + 2 > max_length
        for length in lengths
    )

    report = {
        "num_captions": len(
            captions
        ),
        "min_content_tokens": min(
            lengths
        ),
        "max_content_tokens": max(
            lengths
        ),
        "mean_content_tokens": (
            sum(lengths)
            / len(lengths)
        ),
        "max_encoded_length": (
            max(lengths) + 2
        ),
        "too_long_count": (
            too_long_count
        ),
        "unknown_token_count": sum(
            unknown_counter.values()
        ),
        "unknown_tokens": dict(
            sorted(
                unknown_counter.items()
            )
        ),
    }

    print()
    print(
        f"{name}:"
    )

    print(
        "  captions:",
        report["num_captions"],
    )

    print(
        "  content length:",
        f"{report['min_content_tokens']}"
        f".."
        f"{report['max_content_tokens']}",
    )

    print(
        "  max encoded length:",
        report["max_encoded_length"],
    )

    print(
        "  too long:",
        report["too_long_count"],
    )

    print(
        "  unknown tokens:",
        report["unknown_token_count"],
    )

    return report


def main():
    config = load_config()

    text_config = config[
        "text"
    ]

    max_length = int(
        text_config[
            "max_length"
        ]
    )

    min_frequency = int(
        text_config[
            "min_frequency"
        ]
    )

    train_path = Path(
        config["data"][
            "train_metadata"
        ]
    )

    valid_path = Path(
        config["data"][
            "valid_metadata"
        ]
    )

    test_path = Path(
        config["data"][
            "test_metadata"
        ]
    )

    train_captions = read_captions(
        train_path
    )

    valid_captions = read_captions(
        valid_path
    )

    test_captions = read_captions(
        test_path
    )

    # ---------------------------------
    # CRITICAL:
    # vocabulary is built ONLY
    # from the training split.
    # ---------------------------------

    train_token_sequences = [
        tokenize(
            caption
        )
        for caption in train_captions
    ]

    vocabulary = build_vocabulary(
        token_sequences=(
            train_token_sequences
        ),
        min_frequency=min_frequency,
    )

    vocabulary_path = Path(
        text_config[
            "vocabulary_path"
        ]
    )

    vocabulary.save(
        vocabulary_path
    )

    print(
        "Vocabulary size:",
        len(vocabulary),
    )

    print(
        "Special IDs:",
        {
            "PAD": vocabulary.pad_id,
            "BOS": vocabulary.bos_id,
            "EOS": vocabulary.eos_id,
            "UNK": vocabulary.unk_id,
        },
    )

    print()
    print(
        "Vocabulary:"
    )

    for token_id, token in enumerate(
        vocabulary.id_to_token
    ):
        print(
            f"  {token_id:02d}: "
            f"{token}"
        )

    reports = {
        "train": analyze_split(
            name="train",
            captions=train_captions,
            vocabulary=vocabulary,
            max_length=max_length,
        ),
        "valid": analyze_split(
            name="valid",
            captions=valid_captions,
            vocabulary=vocabulary,
            max_length=max_length,
        ),
        "test": analyze_split(
            name="test",
            captions=test_captions,
            vocabulary=vocabulary,
            max_length=max_length,
        ),
    }

    # No train caption should truncate.
    if reports[
        "train"
    ][
        "too_long_count"
    ] != 0:
        raise RuntimeError(
            "max_length is too small "
            "for training captions"
        )

    if reports[
        "train"
    ][
        "unknown_token_count"
    ] != 0:
        raise RuntimeError(
            "Training vocabulary contains "
            "unexpected unknown tokens"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "vocabulary_size": len(
            vocabulary
        ),
        "min_frequency": (
            min_frequency
        ),
        "max_length": (
            max_length
        ),
        "special_tokens": {
            "<PAD>": (
                vocabulary.pad_id
            ),
            "<BOS>": (
                vocabulary.bos_id
            ),
            "<EOS>": (
                vocabulary.eos_id
            ),
            "<UNK>": (
                vocabulary.unk_id
            ),
        },
        "splits": reports,
    }

    report_path = (
        OUTPUT_ROOT
        / "vocabulary_report.json"
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
        "Saved vocabulary:",
        vocabulary_path,
    )

    print(
        "Saved report:",
        report_path,
    )


if __name__ == "__main__":
    main()