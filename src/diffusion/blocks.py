"""
UNET
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 8,
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

        if time_dim <= 0:
            raise ValueError(
                "time_dim must be positive"
            )

        if (
            in_channels % num_groups != 0
            or out_channels % num_groups != 0
        ):
            raise ValueError(
                "in_channels and out_channels "
                "must be divisible by num_groups"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        self.norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
        )

        self.activation1 = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                out_channels,
            ),
        )

        self.norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
        )

        self.activation2 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]"
            )

        if time_emb.ndim != 2:
            raise ValueError(
                "time_emb must have shape [B, time_dim]"
            )

        if x.shape[0] != time_emb.shape[0]:
            raise ValueError(
                "x and time_emb batch sizes must match"
            )

        if time_emb.shape[1] != self.time_dim:
            raise ValueError(
                f"time_emb feature dimension "
                f"must be {self.time_dim}"
            )

        residual = self.skip(x)

        h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        time_condition = self.time_projection(
            time_emb
        )

        time_condition = time_condition[
            :,
            :,
            None,
            None,
        ]

        h = h + time_condition

        h = self.norm2(h)
        h = self.activation2(h)
        h = self.conv2(h)

        return h + residual


class Downsample(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive"
            )

        self.channels = channels

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"x must have {self.channels} channels"
            )

        height = x.shape[2]
        width = x.shape[3]

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "spatial dimensions must be even"
            )

        return self.conv(x)

class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive"
            )

        self.channels = channels

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"x must have {self.channels} channels"
            )

        x = torch.nn.functional.interpolate(  #interpolate + Conv2d
            x,
            scale_factor=2.0,
            mode="nearest",
        )

        return self.conv(x)