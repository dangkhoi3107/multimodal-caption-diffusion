import torch

from src.diffusion.blocks import (
    Downsample,
    ResidualBlock,
    Upsample,
)


def test_residual_block_same_channels():
    block = ResidualBlock(
        in_channels=64,
        out_channels=64,
        time_dim=256,
    )

    x = torch.randn(
        4,
        64,
        32,
        32,
    )

    time_emb = torch.randn(
        4,
        256,
    )

    output = block(
        x,
        time_emb,
    )

    assert output.shape == (
        4,
        64,
        32,
        32,
    )

    assert torch.isfinite(
        output
    ).all()



def test_residual_block_changes_channels():
    block = ResidualBlock(
        in_channels=64,
        out_channels=128,
        time_dim=256,
    )

    x = torch.randn(
        4,
        64,
        32,
        32,
    )

    time_emb = torch.randn(
        4,
        256,
    )

    output = block(
        x,
        time_emb,
    )

    assert output.shape == (
        4,
        128,
        32,
        32,
    )


def test_residual_block_uses_time_embedding():
    torch.manual_seed(42)

    block = ResidualBlock(
        in_channels=64,
        out_channels=64,
        time_dim=256,
    )

    x = torch.randn(
        2,
        64,
        16,
        16,
    )

    time_zero = torch.zeros(
        2,
        256,
    )

    time_one = torch.ones(
        2,
        256,
    )

    output_zero = block(
        x,
        time_zero,
    )

    output_one = block(
        x,
        time_one,
    )

    assert not torch.allclose(
        output_zero,
        output_one,
    )


def test_residual_block_has_gradients():
    block = ResidualBlock(
        in_channels=64,
        out_channels=64,
        time_dim=256,
    )

    x = torch.randn(
        2,
        64,
        16,
        16,
        requires_grad=True,
    )

    time_emb = torch.randn(
        2,
        256,
        requires_grad=True,
    )

    output = block(
        x,
        time_emb,
    )

    loss = output.mean()

    loss.backward()

    assert x.grad is not None
    assert time_emb.grad is not None

    for parameter in block.parameters():
        assert parameter.grad is not None


def test_downsample_halves_spatial_size():
    block = Downsample(
        channels=64,
    )

    x = torch.randn(
        4,
        64,
        64,
        64,
    )

    output = block(x)

    assert output.shape == (
        4,
        64,
        32,
        32,
    )

    assert torch.isfinite(
        output
    ).all()


def test_upsample_doubles_spatial_size():
    block = Upsample(
        channels=64,
    )

    x = torch.randn(
        4,
        64,
        32,
        32,
    )

    output = block(x)

    assert output.shape == (
        4,
        64,
        64,
        64,
    )

    assert torch.isfinite(
        output
    ).all()


def test_downsample_then_upsample_restores_shape():
    down = Downsample(
        channels=64,
    )

    up = Upsample(
        channels=64,
    )

    x = torch.randn(
        2,
        64,
        64,
        64,
    )

    downsampled = down(x)

    reconstructed_shape = up(
        downsampled
    )

    assert downsampled.shape == (
        2,
        64,
        32,
        32,
    )

    assert reconstructed_shape.shape == (
        2,
        64,
        64,
        64,
    )

def test_downsample_rejects_odd_spatial_size():
    import pytest

    block = Downsample(
        channels=64,
    )

    x = torch.randn(
        2,
        64,
        63,
        64,
    )

    with pytest.raises(
        ValueError,
        match="even",
    ):
        block(x)

def test_downsample_and_upsample_have_gradients():
    down = Downsample(
        channels=64,
    )

    up = Upsample(
        channels=64,
    )

    x = torch.randn(
        2,
        64,
        32,
        32,
        requires_grad=True,
    )

    output = up(
        down(x)
    )

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    for parameter in down.parameters():
        assert parameter.grad is not None

    for parameter in up.parameters():
        assert parameter.grad is not None