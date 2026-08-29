from __future__ import annotations

import torch

from src.captioning.decoder import (
    CaptionDecoder,
    build_causal_attention_mask,
)


VOCAB_SIZE = 19
PAD_ID = 0
MODEL_DIM = 64
MAX_LENGTH = 10


def make_decoder():
    return CaptionDecoder(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        max_length=MAX_LENGTH,
        model_dim=MODEL_DIM,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        dropout=0.0,
    )


def make_inputs(
    batch_size: int = 2,
):
    input_ids = torch.tensor(
        [
            [
                1,
                4,
                15,
                12,
                10,
                14,
                2,
                0,
                0,
                0,
            ],
            [
                1,
                4,
                9,
                8,
                18,
                2,
                0,
                0,
                0,
                0,
            ],
        ],
        dtype=torch.long,
    )[
        :batch_size
    ]

    padding_mask = (
        input_ids
        != PAD_ID
    )

    image_tokens = torch.randn(
        batch_size,
        64,
        MODEL_DIM,
    )

    return (
        input_ids,
        padding_mask,
        image_tokens,
    )


def test_causal_mask_shape_and_values():
    padding_mask = torch.tensor(
        [
            [
                True,
                True,
                True,
                False,
            ]
        ],
        dtype=torch.bool,
    )

    mask = build_causal_attention_mask(
        padding_mask
    )

    assert mask.shape == (
        1,
        1,
        4,
        4,
    )

    expected = torch.tensor(
        [
            [
                [
                    [
                        True,
                        False,
                        False,
                        False,
                    ],
                    [
                        True,
                        True,
                        False,
                        False,
                    ],
                    [
                        True,
                        True,
                        True,
                        False,
                    ],
                    [
                        True,
                        True,
                        True,
                        False,
                    ],
                ]
            ]
        ],
        dtype=torch.bool,
    )

    torch.testing.assert_close(
        mask,
        expected,
    )


def test_decoder_output_shape():
    torch.manual_seed(
        1
    )

    decoder = make_decoder()

    (
        input_ids,
        padding_mask,
        image_tokens,
    ) = make_inputs()

    logits = decoder(
        input_ids=input_ids,
        padding_mask=padding_mask,
        image_tokens=image_tokens,
    )

    assert logits.shape == (
        2,
        MAX_LENGTH,
        VOCAB_SIZE,
    )

    assert torch.isfinite(
        logits
    ).all()


def test_attention_shapes():
    torch.manual_seed(
        2
    )

    decoder = make_decoder()

    (
        input_ids,
        padding_mask,
        image_tokens,
    ) = make_inputs(
        batch_size=1
    )

    (
        logits,
        self_weights,
        cross_weights,
    ) = decoder.forward_with_attention(
        input_ids=input_ids,
        padding_mask=padding_mask,
        image_tokens=image_tokens,
    )

    assert logits.shape == (
        1,
        MAX_LENGTH,
        VOCAB_SIZE,
    )

    assert len(
        self_weights
    ) == 2

    assert len(
        cross_weights
    ) == 2

    assert self_weights[
        0
    ].shape == (
        1,
        4,
        MAX_LENGTH,
        MAX_LENGTH,
    )

    assert cross_weights[
        0
    ].shape == (
        1,
        4,
        MAX_LENGTH,
        64,
    )


def test_future_tokens_do_not_change_prefix_logits():
    torch.manual_seed(
        3
    )

    decoder = make_decoder()
    decoder.eval()

    input_a = torch.tensor(
        [
            [
                1,
                4,
                15,
                12,
                10,
                14,
                2,
                0,
                0,
                0,
            ]
        ],
        dtype=torch.long,
    )

    input_b = input_a.clone()

    # Change future positions only.
    input_b[
        0,
        5,
    ] = 5

    input_b[
        0,
        6,
    ] = 6

    mask_a = (
        input_a
        != PAD_ID
    )

    mask_b = (
        input_b
        != PAD_ID
    )

    image_tokens = torch.randn(
        1,
        64,
        MODEL_DIM,
    )

    with torch.no_grad():
        logits_a = decoder(
            input_ids=input_a,
            padding_mask=mask_a,
            image_tokens=image_tokens,
        )

        logits_b = decoder(
            input_ids=input_b,
            padding_mask=mask_b,
            image_tokens=image_tokens,
        )

    # Positions 0..4 cannot see positions >=5.
    torch.testing.assert_close(
        logits_a[
            :,
            :5,
        ],
        logits_b[
            :,
            :5,
        ],
        rtol=1e-5,
        atol=1e-6,
    )


def test_image_tokens_change_logits():
    torch.manual_seed(
        4
    )

    decoder = make_decoder()
    decoder.eval()

    (
        input_ids,
        padding_mask,
        _,
    ) = make_inputs(
        batch_size=1
    )

    image_a = torch.randn(
        1,
        64,
        MODEL_DIM,
    )

    image_b = torch.randn(
        1,
        64,
        MODEL_DIM,
    )

    with torch.no_grad():
        logits_a = decoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            image_tokens=image_a,
        )

        logits_b = decoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            image_tokens=image_b,
        )

    difference = (
        logits_a
        - logits_b
    ).abs().mean().item()

    assert difference > 0.0


def test_gradient_reaches_text_and_cross_attention():
    torch.manual_seed(
        5
    )

    decoder = make_decoder()

    (
        input_ids,
        padding_mask,
        image_tokens,
    ) = make_inputs(
        batch_size=1
    )

    image_tokens.requires_grad_()

    logits = decoder(
        input_ids=input_ids,
        padding_mask=padding_mask,
        image_tokens=image_tokens,
    )

    loss = (
        logits.square().mean()
    )

    loss.backward()

    assert (
        image_tokens.grad
        is not None
    )

    assert torch.isfinite(
        image_tokens.grad
    ).all()

    assert (
        decoder.token_embedding.weight.grad
        is not None
    )

    assert (
        decoder.layers[
            0
        ].cross_attention.query_projection.weight.grad
        is not None
    )


def test_rejects_non_bool_padding_mask():
    decoder = make_decoder()

    (
        input_ids,
        padding_mask,
        image_tokens,
    ) = make_inputs(
        batch_size=1
    )

    bad_mask = padding_mask.long()

    try:
        decoder(
            input_ids=input_ids,
            padding_mask=bad_mask,
            image_tokens=image_tokens,
        )
    except TypeError:
        return

    raise AssertionError(
        "non-bool padding mask should fail"
    )


def test_rejects_wrong_image_dimension():
    decoder = make_decoder()

    (
        input_ids,
        padding_mask,
        _,
    ) = make_inputs(
        batch_size=1
    )

    image_tokens = torch.randn(
        1,
        64,
        MODEL_DIM + 1,
    )

    try:
        decoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            image_tokens=image_tokens,
        )
    except ValueError:
        return

    raise AssertionError(
        "wrong image-token dimension should fail"
    )


def test_rejects_invalid_first_padding_position():
    decoder = make_decoder()

    (
        input_ids,
        padding_mask,
        image_tokens,
    ) = make_inputs(
        batch_size=1
    )

    padding_mask[
        0,
        0,
    ] = False

    try:
        decoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            image_tokens=image_tokens,
        )
    except ValueError:
        return

    raise AssertionError(
        "position 0 must be valid"
    )
