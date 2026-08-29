from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Shapes:
        query: [B, H, Lq, Dh]
        key:   [B, H, Lk, Dh]
        value: [B, H, Lk, Dh]

        mask:
            bool tensor broadcastable to
            [B, H, Lq, Lk]

            True  = attention allowed
            False = attention blocked

    Returns:
        output:  [B, H, Lq, Dh]
        weights: [B, H, Lq, Lk]
    """

    if query.ndim != 4:
        raise ValueError(
            "query must have shape "
            "[B, H, Lq, Dh]"
        )

    if key.ndim != 4:
        raise ValueError(
            "key must have shape "
            "[B, H, Lk, Dh]"
        )

    if value.ndim != 4:
        raise ValueError(
            "value must have shape "
            "[B, H, Lk, Dh]"
        )

    batch_size = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[3]

    if key.shape[0] != batch_size:
        raise ValueError(
            "query/key batch sizes differ"
        )

    if value.shape[0] != batch_size:
        raise ValueError(
            "query/value batch sizes differ"
        )

    if key.shape[1] != num_heads:
        raise ValueError(
            "query/key head counts differ"
        )

    if value.shape[1] != num_heads:
        raise ValueError(
            "query/value head counts differ"
        )

    if key.shape[3] != head_dim:
        raise ValueError(
            "query/key head dimensions differ"
        )

    if value.shape[3] != head_dim:
        raise ValueError(
            "query/value head dimensions differ"
        )

    if key.shape[2] != value.shape[2]:
        raise ValueError(
            "key/value sequence lengths differ"
        )

    if not torch.is_floating_point(
        query
    ):
        raise TypeError(
            "query must be floating point"
        )

    if (
        query.dtype != key.dtype
        or query.dtype != value.dtype
    ):
        raise TypeError(
            "query/key/value dtypes must match"
        )

    if (
        query.device != key.device
        or query.device != value.device
    ):
        raise ValueError(
            "query/key/value devices must match"
        )

    scale = 1.0 / math.sqrt(
        head_dim
    )

    scores = torch.matmul(
        query,
        key.transpose(
            -2,
            -1,
        ),
    )

    scores = (
        scores * scale
    )

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(
                "attention mask must be bool"
            )

        mask = mask.to(
            device=scores.device
        )

        try:
            expanded_mask = torch.broadcast_to(
                mask,
                scores.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "attention mask is not "
                "broadcastable to attention scores"
            ) from error

        # Every query must have at least
        # one valid key.
        if not expanded_mask.any(
            dim=-1
        ).all():
            raise ValueError(
                "attention mask contains "
                "a fully masked query"
            )

        scores = scores.masked_fill(
            ~expanded_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

    weights = torch.softmax(
        scores,
        dim=-1,
    )

    output = torch.matmul(
        weights,
        value,
    )

    return (
        output,
        weights,
    )


class MultiHeadAttention(
    nn.Module
):
    """Multi-head attention from Linear + matmul.

    Does not use nn.MultiheadAttention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if (
            embedding_dim
            % num_heads
            != 0
        ):
            raise ValueError(
                "embedding_dim must be "
                "divisible by num_heads"
            )

        if not (
            0.0
            <= dropout
            < 1.0
        ):
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        self.embedding_dim = (
            embedding_dim
        )

        self.num_heads = (
            num_heads
        )

        self.head_dim = (
            embedding_dim
            // num_heads
        )

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def split_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """[B, L, D] -> [B, H, L, Dh]."""

        if tensor.ndim != 3:
            raise ValueError(
                "tensor must have shape "
                "[B, L, D]"
            )

        if (
            tensor.shape[-1]
            != self.embedding_dim
        ):
            raise ValueError(
                "last dimension does not "
                "match embedding_dim"
            )

        batch_size = tensor.shape[0]
        sequence_length = tensor.shape[1]

        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return tensor.permute(
            0,
            2,
            1,
            3,
        )

    def merge_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """[B, H, L, Dh] -> [B, L, D]."""

        if tensor.ndim != 4:
            raise ValueError(
                "tensor must have shape "
                "[B, H, L, Dh]"
            )

        if (
            tensor.shape[1]
            != self.num_heads
        ):
            raise ValueError(
                "head count mismatch"
            )

        if (
            tensor.shape[3]
            != self.head_dim
        ):
            raise ValueError(
                "head dimension mismatch"
            )

        batch_size = tensor.shape[0]
        sequence_length = tensor.shape[2]

        tensor = tensor.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        return tensor.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: (
            torch.Tensor | None
        ) = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run multi-head attention.

        Inputs:
            query: [B, Lq, D]
            key:   [B, Lk, D]
            value: [B, Lk, D]

        attention_mask:
            bool, broadcastable to
            [B, H, Lq, Lk]

        Returns:
            output:  [B, Lq, D]
            weights: [B, H, Lq, Lk]
        """

        if (
            query.ndim != 3
            or key.ndim != 3
            or value.ndim != 3
        ):
            raise ValueError(
                "query/key/value must "
                "have shape [B, L, D]"
            )

        if (
            query.shape[0]
            != key.shape[0]
            or query.shape[0]
            != value.shape[0]
        ):
            raise ValueError(
                "batch sizes must match"
            )

        if (
            key.shape[1]
            != value.shape[1]
        ):
            raise ValueError(
                "key/value lengths must match"
            )

        if (
            query.shape[-1]
            != self.embedding_dim
            or key.shape[-1]
            != self.embedding_dim
            or value.shape[-1]
            != self.embedding_dim
        ):
            raise ValueError(
                "embedding dimension mismatch"
            )

        projected_query = (
            self.query_projection(
                query
            )
        )

        projected_key = (
            self.key_projection(
                key
            )
        )

        projected_value = (
            self.value_projection(
                value
            )
        )

        projected_query = (
            self.split_heads(
                projected_query
            )
        )

        projected_key = (
            self.split_heads(
                projected_key
            )
        )

        projected_value = (
            self.split_heads(
                projected_value
            )
        )

        (
            attention_output,
            attention_weights,
        ) = scaled_dot_product_attention(
            query=projected_query,
            key=projected_key,
            value=projected_value,
            mask=attention_mask,
        )

        # Dropout is applied to the
        # attention output for this
        # scratch baseline.
        attention_output = (
            self.dropout(
                attention_output
            )
        )

        merged = self.merge_heads(
            attention_output
        )

        output = (
            self.output_projection(
                merged
            )
        )

        return (
            output,
            attention_weights,
        )