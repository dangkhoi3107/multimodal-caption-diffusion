import torch

from src.text.encoder import (
    SinusoidalPositionalEncoding,
    TextEncoder,
    masked_mean_pool,
)


def make_encoder():
    return TextEncoder(
        vocab_size=19,
        pad_id=0,
        max_length=10,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.0,
    )


def test_positional_encoding_shape():
    encoding = (
        SinusoidalPositionalEncoding(
            embedding_dim=128,
            max_length=10,
        )
    )

    x = torch.zeros(
        2,
        10,
        128,
    )

    output = encoding(
        x
    )

    assert output.shape == (
        2,
        10,
        128,
    )


def test_positions_are_different():
    encoding = (
        SinusoidalPositionalEncoding(
            embedding_dim=128,
            max_length=10,
        )
    )

    x = torch.zeros(
        1,
        10,
        128,
    )

    output = encoding(
        x
    )

    assert not torch.allclose(
        output[:, 0],
        output[:, 1],
    )


def test_masked_mean_pool():
    states = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [100.0, 100.0],
            ]
        ]
    )

    mask = torch.tensor(
        [
            [
                True,
                True,
                False,
            ]
        ]
    )

    pooled = masked_mean_pool(
        states,
        mask,
    )

    expected = torch.tensor(
        [
            [2.0, 3.0]
        ]
    )

    torch.testing.assert_close(
        pooled,
        expected,
    )


def test_text_encoder_shapes():
    encoder = make_encoder()

    token_ids = torch.tensor(
        [
            [
                1, 4, 15, 12, 10,
                14, 2, 0, 0, 0,
            ],
            [
                1, 4, 5, 9, 8,
                17, 2, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    mask = (
        token_ids != 0
    )

    token_states, pooled = (
        encoder(
            token_ids,
            mask,
        )
    )

    assert token_states.shape == (
        2,
        10,
        128,
    )

    assert pooled.shape == (
        2,
        128,
    )


def test_different_text_changes_pool():
    encoder = make_encoder()

    token_ids = torch.tensor(
        [
            [
                1, 4, 15, 12, 10,
                14, 2, 0, 0, 0,
            ],
            [
                1, 4, 5, 9, 8,
                17, 2, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    mask = (
        token_ids != 0
    )

    _, pooled = encoder(
        token_ids,
        mask,
    )

    assert not torch.allclose(
        pooled[0],
        pooled[1],
    )


def test_padding_embedding_does_not_change_pool():
    encoder = make_encoder()

    encoder.eval()

    token_ids = torch.tensor(
        [
            [
                1, 4, 15, 12, 10,
                14, 2, 0, 0, 0,
            ]
        ],
        dtype=torch.long,
    )

    mask = (
        token_ids != 0
    )

    with torch.no_grad():
        _, before = encoder(
            token_ids,
            mask,
        )

        encoder.token_embedding.weight[
            0
        ].fill_(
            1000.0
        )

        _, after = encoder(
            token_ids,
            mask,
        )

    torch.testing.assert_close(
        before,
        after,
        rtol=1e-5,
        atol=1e-6,
    )


def test_gradient_is_finite():
    encoder = make_encoder()

    token_ids = torch.tensor(
        [
            [
                1, 4, 15, 12, 10,
                14, 2, 0, 0, 0,
            ],
            [
                1, 4, 5, 9, 8,
                17, 2, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    mask = (
        token_ids != 0
    )

    _, pooled = encoder(
        token_ids,
        mask,
    )

    loss = (
        pooled ** 2
    ).mean()

    loss.backward()

    for name, parameter in (
        encoder.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        assert (
            parameter.grad
            is not None
        ), name

        assert torch.isfinite(
            parameter.grad
        ).all(), name