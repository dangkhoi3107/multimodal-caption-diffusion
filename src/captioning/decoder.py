from __future__ import annotations

import torch
from torch import nn

from src.text.attention import (
    MultiHeadAttention,
)
from src.text.encoder import (
    FeedForward,
    SinusoidalPositionalEncoding,
)


def build_causal_attention_mask(
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Build decoder self-attention mask.

    Args:
        padding_mask:
            BoolTensor[B, L]
            True = real token
            False = PAD

    Returns:
        BoolTensor[B, 1, L, L]

        A query at position i may attend only to:
        - keys j <= i
        - non-PAD key positions

    Query rows are not removed. This keeps every query with at least
    one valid key as long as BOS is present at position 0.
    """

    if padding_mask.ndim != 2:
        raise ValueError(
            "padding_mask must have shape [B, L]"
        )

    if (
        padding_mask.dtype
        != torch.bool
    ):
        raise TypeError(
            "padding_mask must be bool"
        )

    batch_size = (
        padding_mask.shape[
            0
        ]
    )

    sequence_length = (
        padding_mask.shape[
            1
        ]
    )

    if sequence_length <= 0:
        raise ValueError(
            "sequence length must be positive"
        )

    if not padding_mask[
        :,
        0,
    ].all():
        raise ValueError(
            "decoder position 0 must be a valid token"
        )

    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=(
            padding_mask.device
        ),
    ).tril()

    causal = causal[
        None,
        None,
        :,
        :,
    ]

    valid_keys = padding_mask[
        :,
        None,
        None,
        :,
    ]

    mask = (
        causal
        & valid_keys
    )

    expected_shape = (
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )

    if mask.shape != (
        expected_shape
    ):
        raise ValueError(
            "unexpected causal mask shape"
        )

    return mask


class CaptionDecoderBlock(
    nn.Module
):
    """Pre-norm Transformer decoder block built from scratch.

    Order:
        causal text self-attention
        -> image cross-attention
        -> feed-forward
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if model_dim <= 0:
            raise ValueError(
                "model_dim must be positive"
            )

        if feedforward_dim <= 0:
            raise ValueError(
                "feedforward_dim must be positive"
            )

        self.self_norm = nn.LayerNorm(
            model_dim
        )

        self.self_attention = (
            MultiHeadAttention(
                embedding_dim=(
                    model_dim
                ),
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.cross_norm = nn.LayerNorm(
            model_dim
        )

        self.cross_attention = (
            MultiHeadAttention(
                embedding_dim=(
                    model_dim
                ),
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.feedforward_norm = (
            nn.LayerNorm(
                model_dim
            )
        )

        self.feedforward = FeedForward(
            embedding_dim=model_dim,
            hidden_dim=(
                feedforward_dim
            ),
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        image_tokens: torch.Tensor,
        self_attention_mask: (
            torch.Tensor
        ),
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B,L,D]"
            )

        if image_tokens.ndim != 3:
            raise ValueError(
                "image_tokens must have shape [B,N,D]"
            )

        if (
            x.shape[
                0
            ]
            != image_tokens.shape[
                0
            ]
        ):
            raise ValueError(
                "text/image batch sizes must match"
            )

        if (
            x.shape[
                2
            ]
            != image_tokens.shape[
                2
            ]
        ):
            raise ValueError(
                "text/image model dimensions must match"
            )

        residual = x

        normalized = self.self_norm(
            x
        )

        (
            self_output,
            self_weights,
        ) = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            attention_mask=(
                self_attention_mask
            ),
        )

        x = (
            residual
            + self_output
        )

        residual = x

        normalized = self.cross_norm(
            x
        )

        (
            cross_output,
            cross_weights,
        ) = self.cross_attention(
            query=normalized,
            key=image_tokens,
            value=image_tokens,
            attention_mask=None,
        )

        x = (
            residual
            + cross_output
        )

        residual = x

        normalized = (
            self.feedforward_norm(
                x
            )
        )

        x = (
            residual
            + self.feedforward(
                normalized
            )
        )

        return (
            x,
            self_weights,
            cross_weights,
        )


class CaptionDecoder(
    nn.Module
):
    """Autoregressive Transformer caption decoder.

    Inputs:
        input_ids:    [B, L]
        padding_mask: [B, L], True = real token
        image_tokens: [B, N, D]

    Output:
        logits:       [B, L, vocab_size]

    The decoder is trained using teacher forcing:
        input_ids  = [BOS, w1, ..., wk]
        target_ids = [w1, ..., wk, EOS]
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_length: int,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

        if not (
            0
            <= pad_id
            < vocab_size
        ):
            raise ValueError(
                "pad_id out of range"
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )

        if model_dim <= 0:
            raise ValueError(
                "model_dim must be positive"
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive"
            )

        if (
            model_dim
            % num_heads
            != 0
        ):
            raise ValueError(
                "model_dim must be divisible by num_heads"
            )

        if not (
            0.0
            <= dropout
            < 1.0
        ):
            raise ValueError(
                "dropout must be in [0,1)"
            )

        self.vocab_size = int(
            vocab_size
        )

        self.pad_id = int(
            pad_id
        )

        self.max_length = int(
            max_length
        )

        self.model_dim = int(
            model_dim
        )

        self.token_embedding = nn.Embedding(
            num_embeddings=(
                self.vocab_size
            ),
            embedding_dim=(
                self.model_dim
            ),
            padding_idx=(
                self.pad_id
            ),
        )

        self.position_encoding = (
            SinusoidalPositionalEncoding(
                embedding_dim=(
                    self.model_dim
                ),
                max_length=(
                    self.max_length
                ),
            )
        )

        self.input_dropout = nn.Dropout(
            dropout
        )

        self.layers = nn.ModuleList(
            [
                CaptionDecoderBlock(
                    model_dim=(
                        self.model_dim
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
            self.model_dim
        )

        self.output_projection = (
            nn.Linear(
                self.model_dim,
                self.vocab_size,
                bias=False,
            )
        )

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [B,L]"
            )

        if input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "input_ids must be integer"
            )

        if padding_mask.ndim != 2:
            raise ValueError(
                "padding_mask must have shape [B,L]"
            )

        if (
            padding_mask.dtype
            != torch.bool
        ):
            raise TypeError(
                "padding_mask must be bool"
            )

        if (
            input_ids.shape
            != padding_mask.shape
        ):
            raise ValueError(
                "input_ids/padding_mask shape mismatch"
            )

        if (
            input_ids.shape[
                1
            ]
            > self.max_length
        ):
            raise ValueError(
                "input sequence exceeds max_length"
            )

        if (
            input_ids.min().item()
            < 0
            or input_ids.max().item()
            >= self.vocab_size
        ):
            raise ValueError(
                "input token ID out of range"
            )

        if image_tokens.ndim != 3:
            raise ValueError(
                "image_tokens must have shape [B,N,D]"
            )

        if (
            image_tokens.shape[
                0
            ]
            != input_ids.shape[
                0
            ]
        ):
            raise ValueError(
                "image/text batch sizes differ"
            )

        if (
            image_tokens.shape[
                2
            ]
            != self.model_dim
        ):
            raise ValueError(
                "image token dimension must equal model_dim"
            )

        if not torch.is_floating_point(
            image_tokens
        ):
            raise TypeError(
                "image_tokens must be floating point"
            )

        if (
            input_ids.device
            != padding_mask.device
            or input_ids.device
            != image_tokens.device
        ):
            raise ValueError(
                "decoder inputs must be on the same device"
            )

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        self._validate_inputs(
            input_ids=input_ids,
            padding_mask=padding_mask,
            image_tokens=image_tokens,
        )

        self_attention_mask = (
            build_causal_attention_mask(
                padding_mask
            )
        )

        x = self.token_embedding(
            input_ids
        )

        x = self.position_encoding(
            x
        )

        x = self.input_dropout(
            x
        )

        self_weights_all = []
        cross_weights_all = []

        for layer in self.layers:
            (
                x,
                self_weights,
                cross_weights,
            ) = layer(
                x=x,
                image_tokens=(
                    image_tokens
                ),
                self_attention_mask=(
                    self_attention_mask
                ),
            )

            self_weights_all.append(
                self_weights
            )

            cross_weights_all.append(
                cross_weights
            )

        x = self.final_norm(
            x
        )

        logits = self.output_projection(
            x
        )

        if not torch.isfinite(
            logits
        ).all():
            raise FloatingPointError(
                "decoder produced NaN or Inf"
            )

        return (
            logits,
            self_weights_all,
            cross_weights_all,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        (
            logits,
            _,
            _,
        ) = self.forward_with_attention(
            input_ids=(
                input_ids
            ),
            padding_mask=(
                padding_mask
            ),
            image_tokens=(
                image_tokens
            ),
        )

        return logits
