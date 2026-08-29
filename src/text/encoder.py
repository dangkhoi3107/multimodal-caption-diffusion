from __future__ import annotations

import math

import torch
from torch import nn

from src.text.attention import MultiHeadAttention


class SinusoidalPositionalEncoding(
    nn.Module
):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / embedding_dim
            )
        )

        encoding = torch.zeros(
            max_length,
            embedding_dim,
            dtype=torch.float32,
        )

        encoding[
            :,
            0::2,
        ] = torch.sin(
            position * div_term
        )

        # Odd embedding dimensions need
        # one fewer cosine channel.
        cosine_width = encoding[
            :,
            1::2,
        ].shape[1]

        encoding[
            :,
            1::2,
        ] = torch.cos(
            position
            * div_term[
                :cosine_width
            ]
        )

        self.register_buffer(
            "encoding",
            encoding,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B, L, D]"
            )

        sequence_length = x.shape[1]

        if (
            sequence_length
            > self.encoding.shape[0]
        ):
            raise ValueError(
                "sequence length exceeds "
                "configured max_length"
            )

        positional = self.encoding[
            :sequence_length
        ].to(
            device=x.device,
            dtype=x.dtype,
        )

        return (
            x
            + positional.unsqueeze(0)
        )


class FeedForward(
    nn.Module
):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            x
        )


class TransformerEncoderBlock(
    nn.Module
):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = (
            MultiHeadAttention(
                embedding_dim=(
                    embedding_dim
                ),
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = (
            FeedForward(
                embedding_dim=(
                    embedding_dim
                ),
                hidden_dim=(
                    feedforward_dim
                ),
                dropout=dropout,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if padding_mask.ndim != 2:
            raise ValueError(
                "padding_mask must "
                "have shape [B, L]"
            )

        if padding_mask.dtype != torch.bool:
            raise TypeError(
                "padding_mask must be bool"
            )

        if (
            padding_mask.shape[0]
            != x.shape[0]
            or padding_mask.shape[1]
            != x.shape[1]
        ):
            raise ValueError(
                "padding mask shape mismatch"
            )

        # True = valid key
        attention_mask = (
            padding_mask[
                :,
                None,
                None,
                :,
            ]
        )

        residual = x

        normalized = self.norm1(
            x
        )

        attention_output, _ = (
            self.attention(
                query=normalized,
                key=normalized,
                value=normalized,
                attention_mask=(
                    attention_mask
                ),
            )
        )

        x = (
            residual
            + attention_output
        )

        residual = x

        normalized = self.norm2(
            x
        )

        x = (
            residual
            + self.feed_forward(
                normalized
            )
        )

        return x


def masked_mean_pool(
    token_states: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool valid text tokens only.

    token_states:
        [B, L, D]

    padding_mask:
        [B, L]
        True = valid token

    returns:
        [B, D]
    """

    if token_states.ndim != 3:
        raise ValueError(
            "token_states must have "
            "shape [B, L, D]"
        )

    if padding_mask.ndim != 2:
        raise ValueError(
            "padding_mask must have "
            "shape [B, L]"
        )

    if padding_mask.dtype != torch.bool:
        raise TypeError(
            "padding_mask must be bool"
        )

    if (
        token_states.shape[:2]
        != padding_mask.shape
    ):
        raise ValueError(
            "token_states and "
            "padding_mask shapes mismatch"
        )

    if not padding_mask.any(
        dim=1
    ).all():
        raise ValueError(
            "each sequence must contain "
            "at least one valid token"
        )

    weights = padding_mask.to(
        dtype=token_states.dtype
    ).unsqueeze(
        -1
    )

    summed = (
        token_states
        * weights
    ).sum(
        dim=1
    )

    counts = weights.sum(
        dim=1
    )

    return (
        summed
        / counts
    )


class TextEncoder(
    nn.Module
):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_length: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive"
            )

        if not (
            0 <= pad_id < vocab_size
        ):
            raise ValueError(
                "pad_id out of range"
            )

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.position_encoding = (
            SinusoidalPositionalEncoding(
                embedding_dim=(
                    embedding_dim
                ),
                max_length=max_length,
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=(
                        embedding_dim
                    ),
                    num_heads=num_heads,
                    feedforward_dim=(
                        feedforward_dim
                    ),
                    dropout=dropout,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have "
                "shape [B, L]"
            )

        if token_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "token_ids must be integer"
            )

        if (
            token_ids.shape
            != padding_mask.shape
        ):
            raise ValueError(
                "token_ids and mask "
                "shapes must match"
            )

        if token_ids.shape[1] > (
            self.max_length
        ):
            raise ValueError(
                "sequence exceeds max_length"
            )

        if (
            token_ids.min().item() < 0
            or token_ids.max().item()
            >= self.vocab_size
        ):
            raise ValueError(
                "token ID out of range"
            )

        x = self.token_embedding(
            token_ids
        )

        x = self.position_encoding(
            x
        )

        x = self.dropout(
            x
        )

        for layer in self.layers:
            x = layer(
                x,
                padding_mask,
            )

        token_states = self.final_norm(
            x
        )

        pooled = masked_mean_pool(
            token_states=token_states,
            padding_mask=padding_mask,
        )

        return (
            token_states,
            pooled,
        )