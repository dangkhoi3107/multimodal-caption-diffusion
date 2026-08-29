from __future__ import annotations

import torch
from torch import nn


class TextConditioner(nn.Module):
    """Project pooled text state into U-Net time dimension.

    Input:
        pooled_text: [B, text_dim]

    Output:
        text_condition: [B, condition_dim]
    """

    def __init__(
        self,
        text_dim: int,
        condition_dim: int,
    ) -> None:
        super().__init__()

        if text_dim <= 0:
            raise ValueError(
                "text_dim must be positive"
            )

        if condition_dim <= 0:
            raise ValueError(
                "condition_dim must be positive"
            )

        self.text_dim = text_dim
        self.condition_dim = condition_dim

        self.projection = nn.Sequential(
            nn.Linear(
                text_dim,
                condition_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                condition_dim,
                condition_dim,
            ),
        )

    def forward(
        self,
        pooled_text: torch.Tensor,
    ) -> torch.Tensor:
        if pooled_text.ndim != 2:
            raise ValueError(
                "pooled_text must have "
                "shape [B, D]"
            )

        if (
            pooled_text.shape[1]
            != self.text_dim
        ):
            raise ValueError(
                "pooled_text last dimension "
                f"must be {self.text_dim}"
            )

        if not torch.is_floating_point(
            pooled_text
        ):
            raise TypeError(
                "pooled_text must be floating point"
            )

        output = self.projection(
            pooled_text
        )

        if not torch.isfinite(
            output
        ).all():
            raise ValueError(
                "text condition contains "
                "non-finite values"
            )

        return output


def drop_text_condition(
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    probability: float,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Replace complete prompts with the empty prompt.

    Empty prompt:
        BOS EOS PAD PAD ...

    Args:
        token_ids:
            [B, L]

        padding_mask:
            [B, L], bool
            True = valid token

    Returns:
        output_token_ids:
            [B, L]

        output_padding_mask:
            [B, L]

        dropped:
            [B], bool
    """

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape [B, L]"
        )

    if padding_mask.shape != token_ids.shape:
        raise ValueError(
            "padding_mask shape must "
            "match token_ids"
        )

    if padding_mask.dtype != torch.bool:
        raise TypeError(
            "padding_mask must be bool"
        )

    if token_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "token_ids must have integer dtype"
        )

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise ValueError(
            "probability must be in [0, 1]"
        )

    batch_size = token_ids.shape[0]
    sequence_length = token_ids.shape[1]

    if sequence_length < 2:
        raise ValueError(
            "sequence length must be "
            "at least 2 for BOS/EOS"
        )

    dropped = (
        torch.rand(
            batch_size,
            device=token_ids.device,
            generator=generator,
        )
        < probability
    )

    empty_token_ids = torch.full_like(
        token_ids,
        fill_value=pad_id,
    )

    empty_token_ids[
        :,
        0,
    ] = bos_id

    empty_token_ids[
        :,
        1,
    ] = eos_id

    empty_padding_mask = (
        empty_token_ids
        != pad_id
    )

    output_token_ids = torch.where(
        dropped[:, None],
        empty_token_ids,
        token_ids,
    )

    output_padding_mask = torch.where(
        dropped[:, None],
        empty_padding_mask,
        padding_mask,
    )

    return (
        output_token_ids,
        output_padding_mask,
        dropped,
    )