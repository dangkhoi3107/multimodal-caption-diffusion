from __future__ import annotations

import torch
from torch import nn


def _group_count(
    channels: int,
) -> int:
    """Choose a small valid GroupNorm group count."""
    for groups in (
        8,
        4,
        2,
        1,
    ):
        if (
            channels
            % groups
            == 0
        ):
            return groups

    return 1


class ConvNormAct(
    nn.Module
):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive"
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be positive"
            )

        if stride not in (
            1,
            2,
        ):
            raise ValueError(
                "stride must be 1 or 2"
            )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            num_groups=_group_count(
                out_channels
            ),
            num_channels=out_channels,
        )

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.act(
            self.norm(
                self.conv(
                    x
                )
            )
        )


class ResidualConvBlock(
    nn.Module
):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive"
            )

        self.block1 = ConvNormAct(
            in_channels=channels,
            out_channels=channels,
            stride=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=_group_count(
                channels
            ),
            num_channels=channels,
        )

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.block1(
            x
        )

        x = self.norm2(
            self.conv2(
                x
            )
        )

        return self.act(
            x
            + residual
        )


class ImageEncoder(
    nn.Module
):
    """Small CNN image encoder trained from random initialization.

    For the current Phase 4 baseline:
        input image:      [B, 3, 64, 64]
        feature map:      [B, model_dim, 8, 8]
        spatial tokens:   [B, 64, model_dim]

    The decoder will cross-attend to the 8x8 spatial token grid.
    No pretrained backbone is used.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        model_dim: int = 256,
        image_size: int = 64,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive"
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be positive"
            )

        if model_dim <= 0:
            raise ValueError(
                "model_dim must be positive"
            )

        if image_size <= 0:
            raise ValueError(
                "image_size must be positive"
            )

        if (
            image_size
            % 8
            != 0
        ):
            raise ValueError(
                "image_size must be divisible by 8"
            )

        self.in_channels = int(
            in_channels
        )

        self.base_channels = int(
            base_channels
        )

        self.model_dim = int(
            model_dim
        )

        self.image_size = int(
            image_size
        )

        hidden_1 = (
            self.base_channels
        )

        hidden_2 = (
            self.base_channels
            * 2
        )

        hidden_3 = (
            self.base_channels
            * 4
        )

        # 64 -> 32
        self.stem = ConvNormAct(
            in_channels=(
                self.in_channels
            ),
            out_channels=hidden_1,
            stride=2,
        )

        self.stage1 = (
            ResidualConvBlock(
                hidden_1
            )
        )

        # 32 -> 16
        self.down1 = ConvNormAct(
            in_channels=hidden_1,
            out_channels=hidden_2,
            stride=2,
        )

        self.stage2 = (
            ResidualConvBlock(
                hidden_2
            )
        )

        # 16 -> 8
        self.down2 = ConvNormAct(
            in_channels=hidden_2,
            out_channels=hidden_3,
            stride=2,
        )

        self.stage3 = (
            ResidualConvBlock(
                hidden_3
            )
        )

        self.projection = nn.Conv2d(
            in_channels=hidden_3,
            out_channels=(
                self.model_dim
            ),
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.grid_size = (
            self.image_size
            // 8
        )

        self.num_tokens = (
            self.grid_size
            * self.grid_size
        )

        # Fixed image resolution -> fixed spatial-token count.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens,
                self.model_dim,
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        self.output_norm = nn.LayerNorm(
            self.model_dim
        )

    def forward_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B,C,H,W]"
            )

        if (
            images.shape[
                1
            ]
            != self.in_channels
        ):
            raise ValueError(
                "image channel count does not match encoder"
            )

        if (
            images.shape[
                2
            ]
            != self.image_size
            or images.shape[
                3
            ]
            != self.image_size
        ):
            raise ValueError(
                "image spatial size does not match encoder"
            )

        if not torch.is_floating_point(
            images
        ):
            raise TypeError(
                "images must be floating point"
            )

        x = self.stem(
            images
        )

        x = self.stage1(
            x
        )

        x = self.down1(
            x
        )

        x = self.stage2(
            x
        )

        x = self.down2(
            x
        )

        x = self.stage3(
            x
        )

        x = self.projection(
            x
        )

        expected_shape = (
            images.shape[
                0
            ],
            self.model_dim,
            self.grid_size,
            self.grid_size,
        )

        if x.shape != (
            expected_shape
        ):
            raise ValueError(
                "unexpected CNN feature-map shape: "
                f"expected {expected_shape}, "
                f"got {tuple(x.shape)}"
            )

        return x

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.forward_features(
            images
        )

        # [B, D, H, W]
        # -> [B, H*W, D]
        tokens = (
            features
            .flatten(
                start_dim=2
            )
            .transpose(
                1,
                2,
            )
        )

        if (
            tokens.shape[
                1
            ]
            != self.num_tokens
        ):
            raise ValueError(
                "unexpected spatial token count"
            )

        tokens = (
            tokens
            + self.position_embedding
        )

        tokens = self.output_norm(
            tokens
        )

        if not torch.isfinite(
            tokens
        ).all():
            raise FloatingPointError(
                "image encoder produced NaN or Inf"
            )

        return tokens
