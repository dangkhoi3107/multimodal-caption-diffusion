from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.captioning.metrics import corpus_bleu
from src.captioning.model import CaptionModel
from src.data.caption_dataset import CaptionDataset
from src.text.tokenizer import decode, tokenize
from src.text.vocabulary import Vocabulary


DEFAULT_CHECKPOINT_PATH = Path(
    "checkpoints/image_captioning.pt"
)

DEFAULT_OUTPUT_ROOT = Path(
    "outputs/phase5_visual_conditioning"
)

BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    """Parse reproducible Phase 5 visual-conditioning evaluation paths."""

    parser = argparse.ArgumentParser(
        description="Measure caption sensitivity to real and perturbed visual tokens."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Phase 4 checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for JSON evaluation artifacts.",
    )
    return parser.parse_args()


def restore_vocabulary(
    data: dict,
) -> Vocabulary:
    vocabulary = Vocabulary(
        token_to_id={
            str(token): int(token_id)
            for token, token_id
            in data["token_to_id"].items()
        },
        id_to_token=tuple(
            str(token)
            for token
            in data["id_to_token"]
        ),
    )

    vocabulary.validate()

    return vocabulary


def build_model(
    config: dict,
    vocabulary: Vocabulary,
) -> CaptionModel:
    return CaptionModel(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        image_size=int(
            config["data"]["image_size"]
        ),
        in_channels=int(
            config["model"]["image_encoder"]["in_channels"]
        ),
        base_channels=int(
            config["model"]["image_encoder"]["base_channels"]
        ),
        model_dim=int(
            config["model"]["model_dim"]
        ),
        max_length=int(
            config["text"]["sequence_length"]
        ),
        num_heads=int(
            config["model"]["decoder"]["num_heads"]
        ),
        num_layers=int(
            config["model"]["decoder"]["num_layers"]
        ),
        feedforward_dim=int(
            config["model"]["decoder"]["feedforward_dim"]
        ),
        dropout=float(
            config["model"]["decoder"]["dropout"]
        ),
    )


def build_class_mismatched_permutation(
    class_ids: list[int],
) -> list[int]:
    """Return a one-to-one source permutation with a different class.

    The algorithm sorts samples by class and rotates the source positions
    by the size of the largest class. A mismatch is possible only when no
    class occupies more than half the dataset.
    """

    if not class_ids:
        raise ValueError(
            "class_ids must not be empty"
        )

    counts = Counter(class_ids)
    sample_count = len(class_ids)
    largest_count = max(
        counts.values()
    )

    if largest_count * 2 > sample_count:
        raise ValueError(
            "Cannot build a one-to-one class-mismatched permutation "
            "because one class contains more than half the samples"
        )

    sorted_indices = sorted(
        range(sample_count),
        key=lambda index: (
            class_ids[index],
            index,
        ),
    )

    source_positions = (
        sorted_indices[
            largest_count:
        ]
        + sorted_indices[
            :largest_count
        ]
    )

    permutation = [
        -1
        for _ in range(
            sample_count
        )
    ]

    for (
        target_index,
        source_index,
    ) in zip(
        sorted_indices,
        source_positions,
    ):
        permutation[
            target_index
        ] = source_index

    if sorted(
        permutation
    ) != list(
        range(
            sample_count
        )
    ):
        raise RuntimeError(
            "mismatched mapping is not a permutation"
        )

    for (
        target_index,
        source_index,
    ) in enumerate(
        permutation
    ):
        if (
            class_ids[
                target_index
            ]
            == class_ids[
                source_index
            ]
        ):
            raise RuntimeError(
                "mismatched permutation contains a same-class pair"
            )

    return permutation


@torch.no_grad()
def greedy_generate_from_image_tokens(
    model: CaptionModel,
    image_tokens: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
) -> torch.Tensor:
    if image_tokens.ndim != 3:
        raise ValueError(
            "image_tokens must have shape [B,N,D]"
        )

    if max_length < 2:
        raise ValueError(
            "max_length must be at least 2"
        )

    if max_length > model.max_length:
        raise ValueError(
            "max_length exceeds decoder max_length"
        )

    batch_size = int(
        image_tokens.shape[0]
    )

    generated = torch.full(
        (
            batch_size,
            1,
        ),
        fill_value=bos_id,
        dtype=torch.long,
        device=image_tokens.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=image_tokens.device,
    )

    for _ in range(
        max_length - 1
    ):
        padding_mask = (
            generated
            != pad_id
        )

        logits = model.decoder(
            input_ids=generated,
            padding_mask=padding_mask,
            image_tokens=image_tokens,
        )

        next_token = logits[
            :,
            -1,
        ].argmax(
            dim=-1
        )

        next_token = torch.where(
            finished,
            torch.full_like(
                next_token,
                pad_id,
            ),
            next_token,
        )

        generated = torch.cat(
            [
                generated,
                next_token[
                    :,
                    None,
                ],
            ],
            dim=1,
        )

        finished = (
            finished
            | (
                next_token
                == eos_id
            )
        )

        if finished.all():
            break

    return generated


@torch.no_grad()
def encode_dataset(
    model: CaptionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    list[str],
    list[str],
    list[int],
]:
    token_batches = []
    references = []
    file_names = []
    class_ids = []

    for batch in loader:
        images = batch[
            "image"
        ].to(
            device
        )

        image_tokens = (
            model.encode_images(
                images
            )
        )

        token_batches.append(
            image_tokens.detach().cpu()
        )

        references.extend(
            str(item)
            for item in batch[
                "caption"
            ]
        )

        file_names.extend(
            str(item)
            for item in batch[
                "file_name"
            ]
        )

        class_ids.extend(
            int(item)
            for item in batch[
                "class_id"
            ].tolist()
        )

    return (
        torch.cat(
            token_batches,
            dim=0,
        ),
        references,
        file_names,
        class_ids,
    )


def infer_controlled_caption_class(
    caption: str,
) -> int | None:
    """Infer product class from the controlled Phase 3/4 vocabulary.

    This is a diagnostic for this specific controlled product dataset,
    not a generic image-captioning metric.
    """

    tokens = set(
        tokenize(
            caption
        )
    )

    class_keywords = {
        0: {
            "serum",
        },
        1: {
            "deodorant",
            "tube",
        },
        2: {
            "lifebuoy",
            "handwash",
            "pouch",
        },
    }

    matches = [
        class_id
        for (
            class_id,
            keywords,
        ) in class_keywords.items()
        if tokens.intersection(
            keywords
        )
    ]

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    return None


def evaluate_predictions(
    references: list[str],
    predictions: list[str],
    class_ids: list[int],
) -> dict:
    if not (
        len(
            references
        )
        == len(
            predictions
        )
        == len(
            class_ids
        )
    ):
        raise ValueError(
            "evaluation inputs must have matching lengths"
        )

    reference_tokens = [
        tokenize(
            reference
        )
        for reference
        in references
    ]

    prediction_tokens = [
        tokenize(
            prediction
        )
        for prediction
        in predictions
    ]

    bleu = corpus_bleu(
        references=(
            reference_tokens
        ),
        hypotheses=(
            prediction_tokens
        ),
        max_n=4,
    )

    exact_match = sum(
        int(
            reference
            == prediction
        )
        for (
            reference,
            prediction,
        ) in zip(
            reference_tokens,
            prediction_tokens,
        )
    ) / len(
        references
    )

    inferred_classes = [
        infer_controlled_caption_class(
            prediction
        )
        for prediction
        in predictions
    ]

    class_resolved = sum(
        inferred is not None
        for inferred
        in inferred_classes
    )

    class_correct = sum(
        int(
            inferred
            == target
        )
        for (
            inferred,
            target,
        ) in zip(
            inferred_classes,
            class_ids,
        )
        if inferred is not None
    )

    class_accuracy_resolved = (
        class_correct
        / class_resolved
        if class_resolved > 0
        else 0.0
    )

    class_accuracy_all = (
        class_correct
        / len(
            class_ids
        )
    )

    return {
        "test_samples": len(
            references
        ),
        "exact_match_accuracy": float(
            exact_match
        ),
        **bleu,
        "controlled_class_resolved_rate": float(
            class_resolved
            / len(
                class_ids
            )
        ),
        "controlled_class_accuracy_all": float(
            class_accuracy_all
        ),
        "controlled_class_accuracy_resolved": float(
            class_accuracy_resolved
        ),
    }


@torch.no_grad()
def decode_token_bank(
    model: CaptionModel,
    token_bank: torch.Tensor,
    vocabulary: Vocabulary,
    max_length: int,
    device: torch.device,
) -> list[str]:
    predictions = []

    for start in range(
        0,
        token_bank.shape[
            0
        ],
        BATCH_SIZE,
    ):
        end = min(
            start
            + BATCH_SIZE,
            int(
                token_bank.shape[
                    0
                ]
            ),
        )

        image_tokens = token_bank[
            start:end
        ].to(
            device
        )

        generated = (
            greedy_generate_from_image_tokens(
                model=model,
                image_tokens=image_tokens,
                bos_id=(
                    vocabulary.bos_id
                ),
                eos_id=(
                    vocabulary.eos_id
                ),
                pad_id=(
                    vocabulary.pad_id
                ),
                max_length=(
                    max_length
                ),
            )
        )

        generated = (
            generated
            .detach()
            .cpu()
        )

        predictions.extend(
            decode(
                generated[
                    index
                ],
                vocabulary=(
                    vocabulary
                ),
            )
            for index in range(
                generated.shape[
                    0
                ]
            )
        )

    return predictions


def same_prediction_rate(
    first: list[str],
    second: list[str],
) -> float:
    if len(
        first
    ) != len(
        second
    ):
        raise ValueError(
            "prediction lists must have matching lengths"
        )

    return float(
        sum(
            int(
                left
                == right
            )
            for (
                left,
                right,
            ) in zip(
                first,
                second,
            )
        )
        / len(
            first
        )
    )


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint[
        "config"
    ]

    vocabulary = (
        restore_vocabulary(
            checkpoint[
                "vocabulary"
            ]
        )
    )

    dataset = CaptionDataset(
        metadata_path=Path(
            config[
                "data"
            ][
                "test_metadata"
            ]
        ),
        vocabulary=vocabulary,
        sequence_length=int(
            config[
                "text"
            ][
                "sequence_length"
            ]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        config,
        vocabulary,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    (
        real_tokens,
        references,
        file_names,
        class_ids,
    ) = encode_dataset(
        model=model,
        loader=loader,
        device=device,
    )

    permutation = (
        build_class_mismatched_permutation(
            class_ids
        )
    )

    mismatched_tokens = (
        real_tokens[
            permutation
        ]
        .clone()
    )

    zero_tokens = torch.zeros_like(
        real_tokens
    )

    max_length = int(
        config[
            "generation"
        ][
            "max_length"
        ]
    )

    real_predictions = (
        decode_token_bank(
            model=model,
            token_bank=real_tokens,
            vocabulary=vocabulary,
            max_length=max_length,
            device=device,
        )
    )

    zero_predictions = (
        decode_token_bank(
            model=model,
            token_bank=zero_tokens,
            vocabulary=vocabulary,
            max_length=max_length,
            device=device,
        )
    )

    mismatched_predictions = (
        decode_token_bank(
            model=model,
            token_bank=(
                mismatched_tokens
            ),
            vocabulary=vocabulary,
            max_length=max_length,
            device=device,
        )
    )

    real_metrics = (
        evaluate_predictions(
            references=references,
            predictions=(
                real_predictions
            ),
            class_ids=class_ids,
        )
    )

    zero_metrics = (
        evaluate_predictions(
            references=references,
            predictions=(
                zero_predictions
            ),
            class_ids=class_ids,
        )
    )

    mismatched_metrics = (
        evaluate_predictions(
            references=references,
            predictions=(
                mismatched_predictions
            ),
            class_ids=class_ids,
        )
    )

    comparison = {
        "checkpoint_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "conditions": {
            "real": (
                real_metrics
            ),
            "zero_visual_tokens": (
                zero_metrics
            ),
            "class_mismatched_visual_tokens": (
                mismatched_metrics
            ),
        },
        "delta_vs_real": {
            "zero_visual_tokens": {
                metric: float(
                    zero_metrics[
                        metric
                    ]
                    - real_metrics[
                        metric
                    ]
                )
                for metric in (
                    "exact_match_accuracy",
                    "bleu1",
                    "bleu2",
                    "bleu3",
                    "bleu4",
                    "controlled_class_accuracy_all",
                )
            },
            "class_mismatched_visual_tokens": {
                metric: float(
                    mismatched_metrics[
                        metric
                    ]
                    - real_metrics[
                        metric
                    ]
                )
                for metric in (
                    "exact_match_accuracy",
                    "bleu1",
                    "bleu2",
                    "bleu3",
                    "bleu4",
                    "controlled_class_accuracy_all",
                )
            },
        },
        "prediction_invariance": {
            "zero_same_as_real_rate": (
                same_prediction_rate(
                    real_predictions,
                    zero_predictions,
                )
            ),
            "mismatched_same_as_real_rate": (
                same_prediction_rate(
                    real_predictions,
                    mismatched_predictions,
                )
            ),
        },
    }

    rows = []

    for index in range(
        len(
            references
        )
    ):
        source_index = (
            permutation[
                index
            ]
        )

        rows.append(
            {
                "index": index,
                "file_name": (
                    file_names[
                        index
                    ]
                ),
                "target_class_id": (
                    class_ids[
                        index
                    ]
                ),
                "mismatched_source_index": (
                    source_index
                ),
                "mismatched_source_class_id": (
                    class_ids[
                        source_index
                    ]
                ),
                "reference": (
                    references[
                        index
                    ]
                ),
                "real_prediction": (
                    real_predictions[
                        index
                    ]
                ),
                "zero_prediction": (
                    zero_predictions[
                        index
                    ]
                ),
                "mismatched_prediction": (
                    mismatched_predictions[
                        index
                    ]
                ),
                "zero_changed": (
                    zero_predictions[
                        index
                    ]
                    != real_predictions[
                        index
                    ]
                ),
                "mismatch_changed": (
                    mismatched_predictions[
                        index
                    ]
                    != real_predictions[
                        index
                    ]
                ),
            }
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "comparison.json"
    ).write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        output_root
        / "predictions.json"
    ).write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Checkpoint epoch:",
        checkpoint[
            "epoch"
        ],
    )

    for (
        name,
        metrics,
    ) in (
        (
            "REAL VISUAL TOKENS",
            real_metrics,
        ),
        (
            "ZERO VISUAL TOKENS",
            zero_metrics,
        ),
        (
            "CLASS-MISMATCHED VISUAL TOKENS",
            mismatched_metrics,
        ),
    ):
        print()
        print(name)
        print(
            "Exact:",
            f"{metrics['exact_match_accuracy']:.4f}",
        )
        print(
            "BLEU-1:",
            f"{metrics['bleu1']:.4f}",
        )
        print(
            "BLEU-2:",
            f"{metrics['bleu2']:.4f}",
        )
        print(
            "BLEU-3:",
            f"{metrics['bleu3']:.4f}",
        )
        print(
            "BLEU-4:",
            f"{metrics['bleu4']:.4f}",
        )
        print(
            "Controlled class acc:",
            f"{metrics['controlled_class_accuracy_all']:.4f}",
        )

    print()
    print(
        "DELTA ZERO - REAL"
    )

    for (
        metric,
        value,
    ) in comparison[
        "delta_vs_real"
    ][
        "zero_visual_tokens"
    ].items():
        print(
            f"{metric}: "
            f"{value:+.4f}"
        )

    print()
    print(
        "DELTA MISMATCHED - REAL"
    )

    for (
        metric,
        value,
    ) in comparison[
        "delta_vs_real"
    ][
        "class_mismatched_visual_tokens"
    ].items():
        print(
            f"{metric}: "
            f"{value:+.4f}"
        )

    print()
    print(
        "PREDICTION INVARIANCE"
    )
    print(
        "Zero same as real:",
        f"{comparison['prediction_invariance']['zero_same_as_real_rate']:.4f}",
    )
    print(
        "Mismatched same as real:",
        f"{comparison['prediction_invariance']['mismatched_same_as_real_rate']:.4f}",
    )

    print()
    print(
        "Examples:"
    )

    changed_rows = [
        row
        for row in rows
        if (
            row[
                "zero_changed"
            ]
            or row[
                "mismatch_changed"
            ]
        )
    ]

    for row in changed_rows[
        :12
    ]:
        print(
            f"[target={row['target_class_id']} "
            f"mismatch_src={row['mismatched_source_class_id']}]"
        )
        print(
            "  REF:  ",
            row[
                "reference"
            ],
        )
        print(
            "  REAL: ",
            row[
                "real_prediction"
            ],
        )
        print(
            "  ZERO: ",
            row[
                "zero_prediction"
            ],
        )
        print(
            "  MIS:  ",
            row[
                "mismatched_prediction"
            ],
        )

    print()
    print(
        "Saved:",
        output_root,
    )


if __name__ == "__main__":
    main()
