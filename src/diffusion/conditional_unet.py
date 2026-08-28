import torch

from src.diffusion.conditioning import (
    ClassConditioner,
)
from src.diffusion.embeddings import (
    sinusoidal_timestep_embedding,
)
from src.diffusion.unet import UNet


class ConditionalUNet(UNet):
    """Phase 2 U-Net with class conditioning.

    Condition:
        time_embedding + class_embedding
    """

    def __init__(
        self,
        num_classes: int,
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

        self.num_classes = num_classes
        self.time_dim = time_dim

        self.class_conditioner = (
            ClassConditioner(
                num_classes=num_classes,
                embedding_dim=time_dim,
            )
        )

    @property
    def null_class_id(self) -> int:
        return (
            self.class_conditioner
            .null_class_id
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        # -------------------------
        # Input validation
        # -------------------------

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

        if class_ids.ndim != 1:
            raise ValueError(
                "class_ids must have shape [B]"
            )

        batch_size = x_t.shape[0]

        if timesteps.shape[0] != batch_size:
            raise ValueError(
                "timesteps batch size "
                "must match x_t"
            )

        if class_ids.shape[0] != batch_size:
            raise ValueError(
                "class_ids batch size "
                "must match x_t"
            )

        if (
            x_t.shape[2] % 4 != 0
            or x_t.shape[3] % 4 != 0
        ):
            raise ValueError(
                "spatial dimensions must be "
                "divisible by 4"
            )

        if class_ids.device != x_t.device:
            raise ValueError(
                "class_ids and x_t "
                "must be on the same device"
            )

        if timesteps.device != x_t.device:
            raise ValueError(
                "timesteps and x_t "
                "must be on the same device"
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
        # Class embedding
        # -------------------------

        class_emb = (
            self.class_conditioner(
                class_ids
            )
        )

        # Both have shape:
        # [B, time_dim]
        condition_emb = (
            time_emb
            + class_emb
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

        # -------------------------
        # Middle
        # -------------------------

        h = self.middle_block1(
            h,
            condition_emb,
        )

        h = self.middle_block2(
            h,
            condition_emb,
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
            self.output_conv(
                h
            )
        )

        return predicted_noise