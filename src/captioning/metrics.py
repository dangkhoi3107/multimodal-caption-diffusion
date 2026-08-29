from __future__ import annotations

import math
from collections import Counter


def _ngrams(
    tokens: list[str],
    n: int,
) -> Counter:
    return Counter(
        tuple(tokens[index:index + n])
        for index in range(
            0,
            len(tokens) - n + 1,
        )
    )


def corpus_bleu(
    references: list[list[str]],
    hypotheses: list[list[str]],
    max_n: int = 4,
    smooth: float = 1e-9,
) -> dict:
    """Simple single-reference corpus BLEU-1..BLEU-max_n."""

    if len(references) != len(hypotheses):
        raise ValueError(
            "references/hypotheses length mismatch"
        )

    if not references:
        raise ValueError("empty corpus")

    if max_n <= 0:
        raise ValueError("max_n must be positive")

    clipped = [0] * max_n
    totals = [0] * max_n

    reference_length = 0
    hypothesis_length = 0

    for reference, hypothesis in zip(
        references,
        hypotheses,
    ):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)

        for n in range(1, max_n + 1):
            ref_counts = _ngrams(reference, n)
            hyp_counts = _ngrams(hypothesis, n)

            totals[n - 1] += sum(
                hyp_counts.values()
            )

            clipped[n - 1] += sum(
                min(
                    count,
                    ref_counts.get(ngram, 0),
                )
                for ngram, count
                in hyp_counts.items()
            )

    if hypothesis_length == 0:
        brevity_penalty = 0.0
    elif hypothesis_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(
            1.0
            - (
                reference_length
                / hypothesis_length
            )
        )

    precisions = []

    for correct, total in zip(
        clipped,
        totals,
    ):
        precision = (
            (correct + smooth)
            / (total + smooth)
        )

        precisions.append(
            precision
        )

    scores = {}

    for n in range(1, max_n + 1):
        log_precision = sum(
            math.log(
                max(
                    precision,
                    smooth,
                )
            )
            for precision
            in precisions[:n]
        ) / n

        scores[
            f"bleu{n}"
        ] = float(
            brevity_penalty
            * math.exp(
                log_precision
            )
        )

    scores[
        "brevity_penalty"
    ] = float(
        brevity_penalty
    )

    return scores
