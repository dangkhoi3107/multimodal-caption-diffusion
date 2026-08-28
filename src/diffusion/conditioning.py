from __future__ import annotations

import torch
from torch import nn


class ClassConditioner(nn.Module):
    """Learnable class embeddings for class-conditioned diffusion.

    Real class IDs:
        0 ... num_classes - 1

    Null class ID:
        num_classes

    The null embedding is used for classifier-free guidance.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Reserve one extra embedding for
        # the unconditional/null condition.
        self.null_class_id = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=num_classes + 1,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        if class_ids.ndim != 1:
            raise ValueError(
                "class_ids must have shape [B]"
            )

        if class_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "class_ids must have integer dtype"
            )

        if class_ids.numel() == 0:
            raise ValueError(
                "class_ids must not be empty"
            )

        minimum = int(
            class_ids.min().item()
        )

        maximum = int(
            class_ids.max().item()
        )

        if minimum < 0:
            raise ValueError(
                "class_ids must be non-negative"
            )

        if maximum > self.null_class_id:
            raise ValueError(
                "class_id out of range: "
                f"maximum allowed is "
                f"{self.null_class_id}"
            )

        return self.embedding(
            class_ids
        )


def drop_condition(
    class_ids: torch.Tensor,
    probability: float,
    null_class_id: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly replace real class IDs with the null class ID.

    This is classifier-free guidance condition dropout
    used during training.
    """

    if class_ids.ndim != 1:
        raise ValueError(
            "class_ids must have shape [B]"
        )

    if class_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "class_ids must have integer dtype"
        )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must be in [0, 1]"
        )

    if null_class_id < 0:
        raise ValueError(
            "null_class_id must be non-negative"
        )

    if class_ids.numel() == 0:
        return class_ids.clone()

    if int(class_ids.min().item()) < 0:
        raise ValueError(
            "class_ids must be non-negative"
        )

    # Training input should contain only
    # real class IDs, not the null ID.
    if int(class_ids.max().item()) >= null_class_id:
        raise ValueError(
            "drop_condition expects only "
            "real class IDs"
        )

    if probability == 0.0:
        return class_ids.clone()

    if probability == 1.0:
        return torch.full_like(
            class_ids,
            fill_value=null_class_id,
        )

    random_values = torch.rand(
        class_ids.shape,
        device=class_ids.device,
        generator=generator,
    )

    drop_mask = (
        random_values < probability
    )

    dropped = class_ids.clone()

    dropped[
        drop_mask
    ] = null_class_id

    return dropped