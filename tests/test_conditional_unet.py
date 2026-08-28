import pytest
import torch

from src.diffusion.conditional_unet import (
    ConditionalUNet,
)


def test_conditional_unet_output_shape():
    model = ConditionalUNet(
        num_classes=3,
    )

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [100, 900],
        dtype=torch.long,
    )

    class_ids = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    output = model(
        x_t,
        timesteps,
        class_ids,
    )

    assert output.shape == (
        2,
        3,
        64,
        64,
    )

    assert torch.isfinite(
        output
    ).all()


def test_different_classes_change_output():
    torch.manual_seed(42)

    model = ConditionalUNet(
        num_classes=3,
    )

    model.eval()

    x_t = torch.randn(
        1,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [500],
        dtype=torch.long,
    )

    class_0 = torch.tensor(
        [0],
        dtype=torch.long,
    )

    class_1 = torch.tensor(
        [1],
        dtype=torch.long,
    )

    with torch.no_grad():
        output_0 = model(
            x_t,
            timesteps,
            class_0,
        )

        output_1 = model(
            x_t,
            timesteps,
            class_1,
        )

    assert not torch.allclose(
        output_0,
        output_1,
    )


def test_null_class_forward():
    model = ConditionalUNet(
        num_classes=3,
    )

    assert model.null_class_id == 3

    x_t = torch.randn(
        1,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [300],
        dtype=torch.long,
    )

    class_ids = torch.tensor(
        [3],
        dtype=torch.long,
    )

    output = model(
        x_t,
        timesteps,
        class_ids,
    )

    assert output.shape == (
        1,
        3,
        64,
        64,
    )

    assert torch.isfinite(
        output
    ).all()


def test_gradient_reaches_class_embedding():
    model = ConditionalUNet(
        num_classes=3,
    )

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [100, 700],
        dtype=torch.long,
    )

    class_ids = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    target = torch.randn_like(
        x_t
    )

    output = model(
        x_t,
        timesteps,
        class_ids,
    )

    loss = torch.nn.functional.mse_loss(
        output,
        target,
    )

    loss.backward()

    gradient = (
        model
        .class_conditioner
        .embedding
        .weight
        .grad
    )

    assert gradient is not None

    assert torch.isfinite(
        gradient
    ).all()

    assert gradient.abs().sum() > 0


def test_class_batch_size_mismatch():
    model = ConditionalUNet(
        num_classes=3,
    )

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [100, 200],
        dtype=torch.long,
    )

    class_ids = torch.tensor(
        [0],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError
    ):
        model(
            x_t,
            timesteps,
            class_ids,
        )