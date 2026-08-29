from __future__ import annotations

import torch
import torch.nn.functional as F

from src.captioning.model import CaptionModel


@torch.no_grad()
def greedy_generate(
    model: CaptionModel,
    images: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
) -> torch.Tensor:
    """Greedy autoregressive caption generation.

    Returns:
        LongTensor[B, <=max_length]
        Sequence includes BOS. Generation stops when every sample has
        emitted EOS or max_length is reached.
    """

    if max_length < 2:
        raise ValueError(
            "max_length must be at least 2"
        )

    if max_length > model.max_length:
        raise ValueError(
            "generation max_length exceeds decoder max_length"
        )

    model.eval()

    image_tokens = model.encode_images(
        images
    )

    batch_size = images.shape[0]

    generated = torch.full(
        (batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=images.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=images.device,
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


def _beam_rank_score(
    log_probability: float,
    token_count_without_bos: int,
    length_penalty: float,
) -> float:
    length = max(
        1,
        token_count_without_bos,
    )

    if length_penalty == 0.0:
        return log_probability

    penalty = (
        float(length)
        ** length_penalty
    )

    return (
        log_probability
        / penalty
    )


@torch.no_grad()
def beam_generate(
    model: CaptionModel,
    images: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
    beam_size: int = 3,
    length_penalty: float = 0.6,
) -> torch.Tensor:
    """Beam-search caption generation.

    The implementation intentionally favors clarity over throughput because
    Phase 5 uses it as a controlled decoding ablation on a small test split.

    Returns:
        LongTensor[B, max_length]
        Every sequence begins with BOS and is padded after EOS/termination.
    """

    if images.ndim != 4:
        raise ValueError(
            "images must have shape [B,C,H,W]"
        )

    if max_length < 2:
        raise ValueError(
            "max_length must be at least 2"
        )

    if max_length > model.max_length:
        raise ValueError(
            "generation max_length exceeds decoder max_length"
        )

    if beam_size <= 0:
        raise ValueError(
            "beam_size must be positive"
        )

    if length_penalty < 0.0:
        raise ValueError(
            "length_penalty must be non-negative"
        )

    model.eval()

    all_image_tokens = (
        model.encode_images(
            images
        )
    )

    batch_size = images.shape[0]

    outputs = torch.full(
        (
            batch_size,
            max_length,
        ),
        fill_value=pad_id,
        dtype=torch.long,
        device=images.device,
    )

    for sample_index in range(
        batch_size
    ):
        sample_image_tokens = (
            all_image_tokens[
                sample_index
                :
                sample_index
                + 1
            ]
        )

        # Each beam:
        # (token_list, cumulative_log_probability, finished)
        beams: list[
            tuple[
                list[int],
                float,
                bool,
            ]
        ] = [
            (
                [bos_id],
                0.0,
                False,
            )
        ]

        for _ in range(
            max_length - 1
        ):
            candidates: list[
                tuple[
                    list[int],
                    float,
                    bool,
                ]
            ] = []

            for (
                tokens,
                score,
                finished,
            ) in beams:
                if finished:
                    candidates.append(
                        (
                            tokens,
                            score,
                            True,
                        )
                    )
                    continue

                input_ids = torch.tensor(
                    [
                        tokens
                    ],
                    dtype=torch.long,
                    device=images.device,
                )

                padding_mask = (
                    input_ids
                    != pad_id
                )

                logits = model.decoder(
                    input_ids=input_ids,
                    padding_mask=(
                        padding_mask
                    ),
                    image_tokens=(
                        sample_image_tokens
                    ),
                )

                log_probs = F.log_softmax(
                    logits[
                        0,
                        -1,
                    ],
                    dim=-1,
                )

                top_count = min(
                    beam_size,
                    int(
                        log_probs.shape[
                            0
                        ]
                    ),
                )

                (
                    top_values,
                    top_indices,
                ) = torch.topk(
                    log_probs,
                    k=top_count,
                )

                for (
                    token_log_prob,
                    token_id,
                ) in zip(
                    top_values.tolist(),
                    top_indices.tolist(),
                ):
                    next_tokens = (
                        tokens
                        + [
                            int(
                                token_id
                            )
                        ]
                    )

                    next_score = (
                        score
                        + float(
                            token_log_prob
                        )
                    )

                    next_finished = (
                        int(
                            token_id
                        )
                        == eos_id
                    )

                    candidates.append(
                        (
                            next_tokens,
                            next_score,
                            next_finished,
                        )
                    )

            candidates.sort(
                key=lambda item: (
                    _beam_rank_score(
                        log_probability=(
                            item[
                                1
                            ]
                        ),
                        token_count_without_bos=(
                            len(
                                item[
                                    0
                                ]
                            )
                            - 1
                        ),
                        length_penalty=(
                            length_penalty
                        ),
                    )
                ),
                reverse=True,
            )

            beams = candidates[
                :beam_size
            ]

            if beams and all(
                item[
                    2
                ]
                for item in beams
            ):
                break

        finished_beams = [
            item
            for item in beams
            if item[
                2
            ]
        ]

        final_pool = (
            finished_beams
            if finished_beams
            else beams
        )

        best = max(
            final_pool,
            key=lambda item: (
                _beam_rank_score(
                    log_probability=(
                        item[
                            1
                        ]
                    ),
                    token_count_without_bos=(
                        len(
                            item[
                                0
                            ]
                        )
                        - 1
                    ),
                    length_penalty=(
                        length_penalty
                    ),
                )
            ),
        )

        best_tokens = best[
            0
        ][
            :max_length
        ]

        outputs[
            sample_index,
            :len(
                best_tokens
            ),
        ] = torch.tensor(
            best_tokens,
            dtype=torch.long,
            device=images.device,
        )

    return outputs
