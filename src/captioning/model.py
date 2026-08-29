from __future__ import annotations

import torch
from torch import nn

from src.captioning.decoder import CaptionDecoder
from src.captioning.image_encoder import ImageEncoder


class CaptionModel(nn.Module):
    """End-to-end image captioner trained entirely from scratch."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        image_size: int = 64,
        in_channels: int = 3,
        base_channels: int = 64,
        model_dim: int = 256,
        max_length: int = 10,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.max_length = int(max_length)

        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            model_dim=model_dim,
            image_size=image_size,
        )

        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            max_length=max_length,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.image_encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self.encode_images(images)

        return self.decoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            image_tokens=image_tokens,
        )
