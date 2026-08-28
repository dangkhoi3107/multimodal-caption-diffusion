"""
ref: https://deepwiki.com/wagnchogn/ArtiDiffuser/5.2-timestep-embeddings

"""
import math
from torch import nn
import torch

"""
Nhúng timesteps bằng sóng hình sin
"""
def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:

    # Check 1D
    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps must have shape [B]"
        )

    if dim <= 0:
        raise ValueError(
            "dim must be positive"
        )

    if max_period <= 0:
        raise ValueError(
            "max_period must be positive"
        )

    # input = [SIN(xi), COS(xi)], nên cần //2 chỉ để lấy 1 nửa vòng sin cos
    half_dim = dim // 2

    if half_dim == 0:
        return timesteps.float().unsqueeze(1)

    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / half_dim
    )

    arguments = (
        timesteps.float().unsqueeze(1)
        * frequencies.unsqueeze(0)
    )

    embedding = torch.cat(
        [
            torch.cos(arguments),
            torch.sin(arguments),
        ],
        dim=1,
    )

    if dim % 2 == 1:
        embedding = torch.cat(
            [
                embedding,
                torch.zeros(
                    timesteps.shape[0],
                    1,
                    device=timesteps.device,
                    dtype=embedding.dtype,
                ),
            ],
            dim=1,
        )

    return embedding


class TimestepMLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if time_dim <= 0:
            raise ValueError(
                "time_dim must be positive"
            )

        self.embedding_dim = embedding_dim
        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(
                embedding_dim,
                time_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                time_dim,
                time_dim,
            ),
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError(
                "embedding must have shape [B, D]"
            )

        if (
            embedding.shape[1]
            != self.embedding_dim
        ):
            raise ValueError(
                "embedding feature dimension "
                f"must be {self.embedding_dim}"
            )

        return self.net(
            embedding
        )