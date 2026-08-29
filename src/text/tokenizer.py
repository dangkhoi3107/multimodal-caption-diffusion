from __future__ import annotations

import re

import torch

from src.text.vocabulary import (
    Vocabulary,
)


TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?"
)


def normalize_text(
    text: str,
) -> str:
    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string"
        )

    text = text.lower()
    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def tokenize(
    text: str,
) -> list[str]:
    normalized = normalize_text(
        text
    )

    return TOKEN_PATTERN.findall(
        normalized
    )


def encode(
    text: str,
    vocabulary: Vocabulary,
    max_length: int,
) -> torch.Tensor:
    """Encode one caption.

    Layout:
        BOS tokens EOS PAD...

    Output:
        LongTensor[max_length]
    """

    if max_length < 2:
        raise ValueError(
            "max_length must be at least 2"
        )

    tokens = tokenize(
        text
    )

    # Reserve:
    # 1 position for BOS
    # 1 position for EOS
    max_content_length = (
        max_length - 2
    )

    tokens = tokens[
        :max_content_length
    ]

    token_ids = [
        vocabulary.bos_id,
    ]

    token_ids.extend(
        vocabulary.token_id(
            token
        )
        for token in tokens
    )

    token_ids.append(
        vocabulary.eos_id
    )

    while len(
        token_ids
    ) < max_length:
        token_ids.append(
            vocabulary.pad_id
        )

    return torch.tensor(
        token_ids,
        dtype=torch.long,
    )


def padding_mask(
    token_ids: torch.Tensor,
    vocabulary: Vocabulary,
) -> torch.Tensor:
    """True means real token.

    Input:
        [L] or [B, L]

    Output:
        same shape, bool
    """

    if token_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "token_ids must have integer dtype"
        )

    return (
        token_ids
        != vocabulary.pad_id
    )


def decode(
    token_ids: torch.Tensor
    | list[int],
    vocabulary: Vocabulary,
    stop_at_eos: bool = True,
) -> str:
    if isinstance(
        token_ids,
        torch.Tensor,
    ):
        if token_ids.ndim != 1:
            raise ValueError(
                "decode expects a "
                "1D token sequence"
            )

        ids = (
            token_ids
            .detach()
            .cpu()
            .tolist()
        )

    else:
        ids = list(
            token_ids
        )

    tokens = []

    for token_id in ids:
        token = vocabulary.token(
            int(token_id)
        )

        if (
            token == "<EOS>"
            and stop_at_eos
        ):
            break

        if token in (
            "<PAD>",
            "<BOS>",
            "<EOS>",
        ):
            continue

        tokens.append(
            token
        )

    return " ".join(
        tokens
    )