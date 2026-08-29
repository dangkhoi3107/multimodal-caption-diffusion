from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.captioning.generation import (
    beam_generate,
    greedy_generate,
)
from src.captioning.metrics import (
    corpus_bleu,
)
from src.captioning.model import (
    CaptionModel,
)
from src.data.caption_dataset import (
    CaptionDataset,
)
from src.text.tokenizer import (
    decode,
    tokenize,
)
from src.text.vocabulary import (
    Vocabulary,
)


CHECKPOINT_PATH = Path(
    "outputs/phase4_captioning/best.pt"
)

OUTPUT_ROOT = Path(
    "outputs/phase5_caption_decoding"
)

BEAM_SIZE = 3
LENGTH_PENALTY = 0.6


def restore_vocabulary(
    data: dict,
) -> Vocabulary:
    vocabulary = Vocabulary(
        token_to_id={
            str(token): int(
                token_id
            )
            for (
                token,
                token_id,
            ) in data[
                "token_to_id"
            ].items()
        },
        id_to_token=tuple(
            str(token)
            for token
            in data[
                "id_to_token"
            ]
        ),
    )

    vocabulary.validate()

    return vocabulary


def build_model(
    config: dict,
    vocabulary: Vocabulary,
) -> CaptionModel:
    return CaptionModel(
        vocab_size=len(
            vocabulary
        ),
        pad_id=(
            vocabulary.pad_id
        ),
        image_size=int(
            config[
                "data"
            ][
                "image_size"
            ]
        ),
        in_channels=int(
            config[
                "model"
            ][
                "image_encoder"
            ][
                "in_channels"
            ]
        ),
        base_channels=int(
            config[
                "model"
            ][
                "image_encoder"
            ][
                "base_channels"
            ]
        ),
        model_dim=int(
            config[
                "model"
            ][
                "model_dim"
            ]
        ),
        max_length=int(
            config[
                "text"
            ][
                "sequence_length"
            ]
        ),
        num_heads=int(
            config[
                "model"
            ][
                "decoder"
            ][
                "num_heads"
            ]
        ),
        num_layers=int(
            config[
                "model"
            ][
                "decoder"
            ][
                "num_layers"
            ]
        ),
        feedforward_dim=int(
            config[
                "model"
            ][
                "decoder"
            ][
                "feedforward_dim"
            ]
        ),
        dropout=float(
            config[
                "model"
            ][
                "decoder"
            ][
                "dropout"
            ]
        ),
    )


def sentence_bleu4(
    reference_tokens: list[str],
    hypothesis_tokens: list[str],
) -> float:
    return float(
        corpus_bleu(
            references=[
                reference_tokens
            ],
            hypotheses=[
                hypothesis_tokens
            ],
            max_n=4,
        )[
            "bleu4"
        ]
    )


def analyze_prediction(
    reference: str,
    prediction: str,
) -> dict:
    reference_tokens = tokenize(
        reference
    )

    prediction_tokens = tokenize(
        prediction
    )

    exact = (
        reference_tokens
        == prediction_tokens
    )

    bleu4 = sentence_bleu4(
        reference_tokens,
        prediction_tokens,
    )

    reference_counter = Counter(
        reference_tokens
    )

    prediction_counter = Counter(
        prediction_tokens
    )

    missing_tokens = sorted(
        (
            reference_counter
            - prediction_counter
        ).elements()
    )

    extra_tokens = sorted(
        (
            prediction_counter
            - reference_counter
        ).elements()
    )

    repeated_tokens = sorted(
        token
        for (
            token,
            count,
        ) in prediction_counter.items()
        if count > 1
    )

    if exact or bleu4 >= 0.60:
        quality_bucket = "good"
    elif bleu4 >= 0.25:
        quality_bucket = (
            "average"
        )
    else:
        quality_bucket = "bad"

    return {
        "exact_match": exact,
        "sentence_bleu4": bleu4,
        "quality_bucket": (
            quality_bucket
        ),
        "missing_reference_tokens": (
            missing_tokens
        ),
        "extra_prediction_tokens": (
            extra_tokens
        ),
        "repeated_prediction_tokens": (
            repeated_tokens
        ),
    }


def summarize(
    rows: list[dict],
) -> dict:
    references = [
        tokenize(
            row[
                "reference"
            ]
        )
        for row in rows
    ]

    hypotheses = [
        tokenize(
            row[
                "prediction"
            ]
        )
        for row in rows
    ]

    bleu = corpus_bleu(
        references=references,
        hypotheses=hypotheses,
        max_n=4,
    )

    exact = sum(
        int(
            row[
                "analysis"
            ][
                "exact_match"
            ]
        )
        for row in rows
    ) / len(rows)

    bucket_counts = Counter(
        row[
            "analysis"
        ][
            "quality_bucket"
        ]
        for row in rows
    )

    return {
        "test_samples": len(
            rows
        ),
        "exact_match_accuracy": float(
            exact
        ),
        **bleu,
        "quality_buckets": {
            key: int(
                value
            )
            for (
                key,
                value,
            ) in sorted(
                bucket_counts.items()
            )
        },
    }


def select_examples(
    rows: list[dict],
    per_bucket: int = 5,
) -> dict:
    selected = {}

    for bucket in (
        "good",
        "average",
        "bad",
    ):
        candidates = [
            row
            for row in rows
            if row[
                "analysis"
            ][
                "quality_bucket"
            ]
            == bucket
        ]

        candidates.sort(
            key=lambda row: row[
                "analysis"
            ][
                "sentence_bleu4"
            ],
            reverse=(
                bucket
                != "bad"
            ),
        )

        selected[
            bucket
        ] = candidates[
            :per_bucket
        ]

    return selected


def evaluate_method(
    method_name: str,
    model: CaptionModel,
    loader: DataLoader,
    vocabulary: Vocabulary,
    max_length: int,
    device: torch.device,
) -> tuple[
    dict,
    list[dict],
]:
    rows = []

    for batch in loader:
        images = batch[
            "image"
        ].to(
            device
        )

        if method_name == "greedy":
            generated = (
                greedy_generate(
                    model=model,
                    images=images,
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
        elif method_name == "beam":
            generated = (
                beam_generate(
                    model=model,
                    images=images,
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
                    beam_size=(
                        BEAM_SIZE
                    ),
                    length_penalty=(
                        LENGTH_PENALTY
                    ),
                )
            )
        else:
            raise ValueError(
                f"unknown decoding method: {method_name}"
            )

        generated = (
            generated
            .detach()
            .cpu()
        )

        for index in range(
            images.shape[
                0
            ]
        ):
            reference = str(
                batch[
                    "caption"
                ][
                    index
                ]
            )

            prediction = decode(
                generated[
                    index
                ],
                vocabulary=(
                    vocabulary
                ),
            )

            analysis = (
                analyze_prediction(
                    reference=reference,
                    prediction=prediction,
                )
            )

            rows.append(
                {
                    "file_name": str(
                        batch[
                            "file_name"
                        ][
                            index
                        ]
                    ),
                    "class_id": int(
                        batch[
                            "class_id"
                        ][
                            index
                        ].item()
                    ),
                    "reference": (
                        reference
                    ),
                    "prediction": (
                        prediction
                    ),
                    "analysis": (
                        analysis
                    ),
                }
            )

    summary = summarize(
        rows
    )

    return (
        summary,
        rows,
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
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
        batch_size=int(
            config[
                "training"
            ][
                "batch_size"
            ]
        ),
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

    max_length = int(
        config[
            "generation"
        ][
            "max_length"
        ]
    )

    (
        greedy_summary,
        greedy_rows,
    ) = evaluate_method(
        method_name="greedy",
        model=model,
        loader=loader,
        vocabulary=vocabulary,
        max_length=max_length,
        device=device,
    )

    (
        beam_summary,
        beam_rows,
    ) = evaluate_method(
        method_name="beam",
        model=model,
        loader=loader,
        vocabulary=vocabulary,
        max_length=max_length,
        device=device,
    )

    comparison = {
        "checkpoint_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "beam_size": (
            BEAM_SIZE
        ),
        "length_penalty": (
            LENGTH_PENALTY
        ),
        "greedy": (
            greedy_summary
        ),
        "beam": (
            beam_summary
        ),
        "delta_beam_minus_greedy": {
            metric: float(
                beam_summary[
                    metric
                ]
                - greedy_summary[
                    metric
                ]
            )
            for metric in (
                "exact_match_accuracy",
                "bleu1",
                "bleu2",
                "bleu3",
                "bleu4",
            )
        },
    }

    qualitative = {
        "greedy": (
            select_examples(
                greedy_rows
            )
        ),
        "beam": (
            select_examples(
                beam_rows
            )
        ),
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_ROOT
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
        OUTPUT_ROOT
        / "greedy_predictions.json"
    ).write_text(
        json.dumps(
            greedy_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_ROOT
        / "beam_predictions.json"
    ).write_text(
        json.dumps(
            beam_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_ROOT
        / "qualitative_examples.json"
    ).write_text(
        json.dumps(
            qualitative,
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

    print()
    print("GREEDY")
    print(
        "Exact:",
        f"{greedy_summary['exact_match_accuracy']:.4f}",
    )
    print(
        "BLEU-1:",
        f"{greedy_summary['bleu1']:.4f}",
    )
    print(
        "BLEU-2:",
        f"{greedy_summary['bleu2']:.4f}",
    )
    print(
        "BLEU-3:",
        f"{greedy_summary['bleu3']:.4f}",
    )
    print(
        "BLEU-4:",
        f"{greedy_summary['bleu4']:.4f}",
    )
    print(
        "Buckets:",
        greedy_summary[
            "quality_buckets"
        ],
    )

    print()
    print(
        f"BEAM size={BEAM_SIZE} "
        f"length_penalty={LENGTH_PENALTY}"
    )
    print(
        "Exact:",
        f"{beam_summary['exact_match_accuracy']:.4f}",
    )
    print(
        "BLEU-1:",
        f"{beam_summary['bleu1']:.4f}",
    )
    print(
        "BLEU-2:",
        f"{beam_summary['bleu2']:.4f}",
    )
    print(
        "BLEU-3:",
        f"{beam_summary['bleu3']:.4f}",
    )
    print(
        "BLEU-4:",
        f"{beam_summary['bleu4']:.4f}",
    )
    print(
        "Buckets:",
        beam_summary[
            "quality_buckets"
        ],
    )

    print()
    print(
        "DELTA beam - greedy"
    )

    for (
        metric,
        value,
    ) in comparison[
        "delta_beam_minus_greedy"
    ].items():
        print(
            f"{metric}: "
            f"{value:+.4f}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
