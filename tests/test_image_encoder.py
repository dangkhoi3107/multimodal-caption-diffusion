from __future__ import annotations

import torch

from src.captioning.image_encoder import (
    ImageEncoder,
)


def make_model():
    return ImageEncoder(
        in_channels=3,
        base_channels=32,
        model_dim=128,
        image_size=64,
    )


def test_image_encoder_output_shape():
    model = make_model()

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    tokens = model(
        images
    )

    assert tokens.shape == (
        2,
        64,
        128,
    )

    assert torch.isfinite(
        tokens
    ).all()


def test_forward_feature_map_shape():
    model = make_model()

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    features = model.forward_features(
        images
    )

    assert features.shape == (
        2,
        128,
        8,
        8,
    )


def test_different_images_change_tokens():
    torch.manual_seed(
        123
    )

    model = make_model()
    model.eval()

    image_a = torch.zeros(
        1,
        3,
        64,
        64,
    )

    image_b = torch.ones(
        1,
        3,
        64,
        64,
    )

    with torch.no_grad():
        tokens_a = model(
            image_a
        )

        tokens_b = model(
            image_b
        )

    difference = (
        tokens_a
        - tokens_b
    ).abs().mean().item()

    assert difference > 0.0


def test_gradient_reaches_parameters():
    model = make_model()

    images = torch.randn(
        2,
        3,
        64,
        64,
        requires_grad=True,
    )

    tokens = model(
        images
    )

    loss = (
        tokens.square().mean()
    )

    loss.backward()

    assert (
        images.grad
        is not None
    )

    assert torch.isfinite(
        images.grad
    ).all()

    found_parameter_gradient = False

    for parameter in (
        model.parameters()
    ):
        if parameter.grad is None:
            continue

        found_parameter_gradient = True

        assert torch.isfinite(
            parameter.grad
        ).all()

    assert found_parameter_gradient


def test_position_embedding_shape():
    model = make_model()

    assert (
        model.position_embedding.shape
        == (
            1,
            64,
            128,
        )
    )

    assert (
        model.num_tokens
        == 64
    )

    assert (
        model.grid_size
        == 8
    )


def test_rejects_wrong_spatial_size():
    model = make_model()

    images = torch.randn(
        1,
        3,
        32,
        32,
    )

    try:
        model(
            images
        )
    except ValueError:
        return

    raise AssertionError(
        "wrong image size should fail"
    )


def test_rejects_wrong_channel_count():
    model = make_model()

    images = torch.randn(
        1,
        1,
        64,
        64,
    )

    try:
        model(
            images
        )
    except ValueError:
        return

    raise AssertionError(
        "wrong channel count should fail"
    )


def test_rejects_integer_images():
    model = make_model()

    images = torch.zeros(
        1,
        3,
        64,
        64,
        dtype=torch.uint8,
    )

    try:
        model(
            images
        )
    except TypeError:
        return

    raise AssertionError(
        "integer images should fail"
    )


def test_batch_independence_in_eval_mode():
    torch.manual_seed(
        7
    )

    model = make_model()
    model.eval()

    image = torch.randn(
        1,
        3,
        64,
        64,
    )

    other = torch.randn(
        1,
        3,
        64,
        64,
    )

    with torch.no_grad():
        single = model(
            image
        )

        batched = model(
            torch.cat(
                [
                    image,
                    other,
                ],
                dim=0,
            )
        )[
            :1
        ]

    torch.testing.assert_close(
        single,
        batched,
        rtol=1e-5,
        atol=1e-5,
    )
