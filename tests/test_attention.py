import pytest
import torch

from src.text.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)


def test_scaled_attention_shapes():
    query = torch.randn(
        2,
        4,
        5,
        32,
    )

    key = torch.randn(
        2,
        4,
        7,
        32,
    )

    value = torch.randn(
        2,
        4,
        7,
        32,
    )

    output, weights = (
        scaled_dot_product_attention(
            query,
            key,
            value,
        )
    )

    assert output.shape == (
        2,
        4,
        5,
        32,
    )

    assert weights.shape == (
        2,
        4,
        5,
        7,
    )


def test_attention_weights_sum_to_one():
    query = torch.randn(
        2,
        4,
        5,
        32,
    )

    key = torch.randn(
        2,
        4,
        7,
        32,
    )

    value = torch.randn(
        2,
        4,
        7,
        32,
    )

    _, weights = (
        scaled_dot_product_attention(
            query,
            key,
            value,
        )
    )

    sums = weights.sum(
        dim=-1
    )

    torch.testing.assert_close(
        sums,
        torch.ones_like(
            sums
        ),
    )


def test_attention_mask_blocks_keys():
    query = torch.randn(
        1,
        2,
        3,
        8,
    )

    key = torch.randn(
        1,
        2,
        4,
        8,
    )

    value = torch.randn(
        1,
        2,
        4,
        8,
    )

    # [B, 1, 1, Lk]
    mask = torch.tensor(
        [
            [
                [
                    True,
                    True,
                    False,
                    False,
                ]
            ]
        ],
        dtype=torch.bool,
    )

    _, weights = (
        scaled_dot_product_attention(
            query,
            key,
            value,
            mask=mask,
        )
    )

    assert torch.all(
        weights[
            ...,
            2:
        ] == 0
    )


def test_fully_masked_query_rejected():
    query = torch.randn(
        1,
        1,
        2,
        8,
    )

    key = torch.randn(
        1,
        1,
        3,
        8,
    )

    value = torch.randn(
        1,
        1,
        3,
        8,
    )

    mask = torch.zeros(
        1,
        1,
        2,
        3,
        dtype=torch.bool,
    )

    with pytest.raises(
        ValueError
    ):
        scaled_dot_product_attention(
            query,
            key,
            value,
            mask=mask,
        )


def test_split_merge_round_trip():
    attention = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    tensor = torch.randn(
        2,
        10,
        128,
    )

    split = attention.split_heads(
        tensor
    )

    assert split.shape == (
        2,
        4,
        10,
        32,
    )

    merged = attention.merge_heads(
        split
    )

    torch.testing.assert_close(
        merged,
        tensor,
    )


def test_multihead_output_shape():
    attention = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    x = torch.randn(
        2,
        10,
        128,
    )

    output, weights = attention(
        x,
        x,
        x,
    )

    assert output.shape == (
        2,
        10,
        128,
    )

    assert weights.shape == (
        2,
        4,
        10,
        10,
    )

    assert torch.isfinite(
        output
    ).all()


def test_padding_mask_broadcast():
    attention = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    x = torch.randn(
        2,
        5,
        128,
    )

    # True = valid key.
    padding_mask = torch.tensor(
        [
            [
                True,
                True,
                True,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                False,
            ],
        ],
        dtype=torch.bool,
    )

    # [B, 1, 1, L]
    attention_mask = (
        padding_mask[
            :,
            None,
            None,
            :,
        ]
    )

    _, weights = attention(
        x,
        x,
        x,
        attention_mask=(
            attention_mask
        ),
    )

    assert torch.all(
        weights[
            0,
            ...,
            3:
        ] == 0
    )

    assert torch.all(
        weights[
            1,
            ...,
            4:
        ] == 0
    )


def test_gradient_is_finite():
    attention = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    x = torch.randn(
        2,
        6,
        128,
        requires_grad=True,
    )

    output, _ = attention(
        x,
        x,
        x,
    )

    loss = (
        output ** 2
    ).mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()

    for parameter in (
        attention.parameters()
    ):
        assert (
            parameter.grad
            is not None
        )

        assert torch.isfinite(
            parameter.grad
        ).all()


def test_embedding_dim_must_divide_heads():
    with pytest.raises(
        ValueError
    ):
        MultiHeadAttention(
            embedding_dim=130,
            num_heads=4,
        )