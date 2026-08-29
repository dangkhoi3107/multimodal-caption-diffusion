from __future__ import annotations

import torch

from src.diffusion.embeddings import (
    sinusoidal_timestep_embedding,
)
from src.diffusion.text_conditioning import (
    TextConditioner,
)
from src.diffusion.unet import UNet
from src.text.encoder import TextEncoder


class TextConditionalUNet(UNet):
    """Phase 3 U-Net with pooled text conditioning.

    text:
        token_ids
        -> TextEncoder
        -> pooled_text [B, text_dim]
        -> TextConditioner
        -> [B, time_dim]

    final conditioning:
        time_embedding + text_condition
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_length: int,
        text_embedding_dim: int = 128,
        text_num_heads: int = 4,
        text_num_layers: int = 2,
        text_feedforward_dim: int = 256,
        text_dropout: float = 0.1,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
        time_dim: int = 256,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
            time_dim=time_dim,
        )

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_length = max_length
        self.time_dim = time_dim

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            max_length=max_length,
            embedding_dim=(
                text_embedding_dim
            ),
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            feedforward_dim=(
                text_feedforward_dim
            ),
            dropout=text_dropout,
        )

        self.text_conditioner = (
            TextConditioner(
                text_dim=(
                    text_embedding_dim
                ),
                condition_dim=time_dim,
            )
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ---------------------------------
        # Input validation
        # ---------------------------------

        if x_t.ndim != 4:
            raise ValueError(
                "x_t must have shape "
                "[B, C, H, W]"
            )

        if (
            x_t.shape[1]
            != self.in_channels
        ):
            raise ValueError(
                f"x_t must have "
                f"{self.in_channels} channels"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have "
                "shape [B]"
            )

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have "
                "shape [B, L]"
            )

        if padding_mask.ndim != 2:
            raise ValueError(
                "padding_mask must have "
                "shape [B, L]"
            )

        if (
            token_ids.shape
            != padding_mask.shape
        ):
            raise ValueError(
                "token_ids and padding_mask "
                "must have identical shape"
            )

        batch_size = x_t.shape[0]

        if (
            timesteps.shape[0]
            != batch_size
        ):
            raise ValueError(
                "timesteps batch size "
                "must match x_t"
            )

        if (
            token_ids.shape[0]
            != batch_size
        ):
            raise ValueError(
                "token_ids batch size "
                "must match x_t"
            )

        if (
            token_ids.shape[1]
            > self.max_length
        ):
            raise ValueError(
                "text sequence exceeds "
                "max_length"
            )

        if (
            x_t.shape[2] % 4 != 0
            or x_t.shape[3] % 4 != 0
        ):
            raise ValueError(
                "spatial dimensions must "
                "be divisible by 4"
            )

        if (
            timesteps.device
            != x_t.device
        ):
            raise ValueError(
                "timesteps and x_t must "
                "be on same device"
            )

        if (
            token_ids.device
            != x_t.device
        ):
            raise ValueError(
                "token_ids and x_t must "
                "be on same device"
            )

        if (
            padding_mask.device
            != x_t.device
        ):
            raise ValueError(
                "padding_mask and x_t must "
                "be on same device"
            )

        # ---------------------------------
        # Timestep condition
        # ---------------------------------

        time_emb = (
            sinusoidal_timestep_embedding(
                timesteps=timesteps,
                dim=(
                    self.time_embedding_dim
                ),
            )
        )

        time_emb = self.time_mlp(
            time_emb
        )

        # ---------------------------------
        # Text condition
        # ---------------------------------

        (
            _,
            pooled_text,
        ) = self.text_encoder(
            token_ids=token_ids,
            padding_mask=padding_mask,
        )

        text_emb = (
            self.text_conditioner(
                pooled_text
            )
        )

        # Both:
        # [B, time_dim]
        condition_emb = (
            time_emb
            + text_emb
        )

        # ---------------------------------
        # U-Net encoder
        # ---------------------------------

        h = self.input_conv(
            x_t
        )

        h = self.down_block1(
            h,
            condition_emb,
        )

        skip1 = h

        h = self.downsample1(
            h
        )

        h = self.down_block2(
            h,
            condition_emb,
        )

        skip2 = h

        h = self.downsample2(
            h
        )

        # ---------------------------------
        # Middle
        # ---------------------------------

        h = self.middle_block1(
            h,
            condition_emb,
        )

        h = self.middle_block2(
            h,
            condition_emb,
        )

        # ---------------------------------
        # Decoder
        # ---------------------------------

        h = self.upsample1(
            h
        )

        h = torch.cat(
            [
                h,
                skip2,
            ],
            dim=1,
        )

        h = self.up_block1(
            h,
            condition_emb,
        )

        h = self.upsample2(
            h
        )

        h = torch.cat(
            [
                h,
                skip1,
            ],
            dim=1,
        )

        h = self.up_block2(
            h,
            condition_emb,
        )

        # ---------------------------------
        # Output
        # ---------------------------------

        h = self.output_norm(
            h
        )

        h = self.output_activation(
            h
        )

        return self.output_conv(
            h
        )