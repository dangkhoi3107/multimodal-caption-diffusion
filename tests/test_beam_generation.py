from __future__ import annotations

import torch
from torch import nn

from src.captioning.generation import (
    beam_generate,
)


class DummyDecoder(
    nn.Module
):
    def __init__(
        self,
        vocab_size: int = 6,
    ) -> None:
        super().__init__()
        self.vocab_size = (
            vocab_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = (
            input_ids.shape[
                0
            ]
        )

        sequence_length = (
            input_ids.shape[
                1
            ]
        )

        logits = torch.full(
            (
                batch_size,
                sequence_length,
                self.vocab_size,
            ),
            fill_value=-10.0,
            device=input_ids.device,
        )

        # BOS -> token 4
        if sequence_length == 1:
            logits[
                :,
                -1,
                4,
            ] = 10.0

            logits[
                :,
                -1,
                5,
            ] = 8.0

        # token 4 -> EOS
        else:
            logits[
                :,
                -1,
                2,
            ] = 10.0

            logits[
                :,
                -1,
                5,
            ] = 8.0

        return logits


class DummyCaptionModel(
    nn.Module
):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.max_length = 6
        self.decoder = (
            DummyDecoder()
        )

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            images.shape[
                0
            ],
            4,
            8,
            device=images.device,
        )


def test_beam_generation_known_path():
    model = (
        DummyCaptionModel()
    )

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    generated = beam_generate(
        model=model,
        images=images,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_length=6,
        beam_size=3,
        length_penalty=0.6,
    )

    assert generated.shape == (
        2,
        6,
    )

    expected_prefix = torch.tensor(
        [
            1,
            4,
            2,
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        generated[
            0,
            :3,
        ],
        expected_prefix,
    )

    torch.testing.assert_close(
        generated[
            1,
            :3,
        ],
        expected_prefix,
    )

    assert (
        generated[
            :,
            3:
        ]
        == 0
    ).all()


def test_beam_generation_is_deterministic():
    model = (
        DummyCaptionModel()
    )

    images = torch.randn(
        1,
        3,
        64,
        64,
    )

    first = beam_generate(
        model=model,
        images=images,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_length=6,
        beam_size=3,
        length_penalty=0.6,
    )

    second = beam_generate(
        model=model,
        images=images,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_length=6,
        beam_size=3,
        length_penalty=0.6,
    )

    torch.testing.assert_close(
        first,
        second,
    )


def test_beam_rejects_invalid_beam_size():
    model = (
        DummyCaptionModel()
    )

    images = torch.randn(
        1,
        3,
        64,
        64,
    )

    try:
        beam_generate(
            model=model,
            images=images,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            max_length=6,
            beam_size=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "beam_size=0 should fail"
    )
