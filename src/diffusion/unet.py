import torch
from torch import nn

from src.diffusion.blocks import (
    Downsample,
    ResidualBlock,
    Upsample,
)
from src.diffusion.embeddings import (
    TimestepMLP,
    sinusoidal_timestep_embedding,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
        time_dim: int = 256,
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

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be positive"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embedding_dim = (
            time_embedding_dim
        )

        self.time_mlp = TimestepMLP(
            embedding_dim=time_embedding_dim,
            time_dim=time_dim,
        )

        # -------------------------
        # Input
        # -------------------------

        self.input_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # -------------------------
        # Down path
        # -------------------------

        self.down_block1 = ResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            time_dim=time_dim,
        )

        self.downsample1 = Downsample(
            channels=base_channels,
        )

        self.down_block2 = ResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            time_dim=time_dim,
        )

        self.downsample2 = Downsample(
            channels=base_channels * 2,
        )

        # -------------------------
        # Middle
        # -------------------------

        self.middle_block1 = ResidualBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            time_dim=time_dim,
        )

        self.middle_block2 = ResidualBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            time_dim=time_dim,
        )

        # -------------------------
        # Up path
        # -------------------------

        self.upsample1 = Upsample(
            channels=base_channels * 4,
        )

        self.up_block1 = ResidualBlock(
            in_channels=base_channels * 6,
            out_channels=base_channels * 2,
            time_dim=time_dim,
        )

        self.upsample2 = Upsample(
            channels=base_channels * 2,
        )

        self.up_block2 = ResidualBlock(
            in_channels=base_channels * 3,
            out_channels=base_channels,
            time_dim=time_dim,
        )

        # -------------------------
        # Output
        # -------------------------

        self.output_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=base_channels,
        )

        self.output_activation = nn.SiLU()

        self.output_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(
                "x_t must have shape [B, C, H, W]"
            )

        if x_t.shape[1] != self.in_channels:
            raise ValueError(
                f"x_t must have "
                f"{self.in_channels} channels"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [B]"
            )

        if timesteps.shape[0] != x_t.shape[0]:
            raise ValueError(
                "timesteps batch size must match x_t"
            )

        if (
            x_t.shape[2] % 4 != 0
            or x_t.shape[3] % 4 != 0
        ):
            raise ValueError(
                "spatial dimensions must be "
                "divisible by 4"
            )

        # -------------------------
        # Timestep embedding
        # -------------------------

        time_emb = (
            sinusoidal_timestep_embedding(
                timesteps=timesteps,
                dim=self.time_embedding_dim,
            )
        )

        time_emb = self.time_mlp(
            time_emb
        )

        # -------------------------
        # Input
        # -------------------------

        h = self.input_conv(
            x_t
        )

        # -------------------------
        # Encoder
        # -------------------------

        h = self.down_block1(
            h,
            time_emb,
        )

        skip1 = h

        h = self.downsample1(
            h
        )

        h = self.down_block2(
            h,
            time_emb,
        )

        skip2 = h

        h = self.downsample2(
            h
        )

        # -------------------------
        # Middle
        # -------------------------

        h = self.middle_block1(
            h,
            time_emb,
        )

        h = self.middle_block2(
            h,
            time_emb,
        )

        # -------------------------
        # Decoder
        # -------------------------

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
            time_emb,
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
            time_emb,
        )

        # -------------------------
        # Output
        # -------------------------

        h = self.output_norm(
            h
        )

        h = self.output_activation(
            h
        )

        predicted_noise = (
            self.output_conv(h)
        )

        return predicted_noise