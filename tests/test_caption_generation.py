from __future__ import annotations

import torch

from src.captioning.generation import (
    greedy_generate,
)
from src.captioning.model import (
    CaptionModel,
)


def test_greedy_generation_contract():
    torch.manual_seed(1)

    model = CaptionModel(
        vocab_size=19,
        pad_id=0,
        image_size=64,
        in_channels=3,
        base_channels=16,
        model_dim=64,
        max_length=10,
        num_heads=4,
        num_layers=1,
        feedforward_dim=128,
        dropout=0.0,
    )

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    generated = greedy_generate(
        model=model,
        images=images,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_length=10,
    )

    assert generated.ndim == 2
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 10
    assert generated.shape[1] >= 2
    assert (
        generated[:, 0]
        == 1
    ).all()
