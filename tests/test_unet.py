import torch

from src.diffusion.unet import (
    UNet,
)


def test_unet_output_shape():
    model = UNet()

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

    output = model(
        x_t,
        timesteps,
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


def test_unet_uses_timesteps():
    torch.manual_seed(42)

    model = UNet()

    x_t = torch.randn(
        1,
        3,
        64,
        64,
    )

    timestep_early = torch.tensor(
        [10],
        dtype=torch.long,
    )

    timestep_late = torch.tensor(
        [900],
        dtype=torch.long,
    )

    output_early = model(
        x_t,
        timestep_early,
    )

    output_late = model(
        x_t,
        timestep_late,
    )

    assert not torch.allclose(
        output_early,
        output_late,
    )


def test_unet_backward_has_finite_gradients():
    model = UNet()

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

    target_noise = torch.randn_like(
        x_t
    )

    predicted_noise = model(
        x_t,
        timesteps,
    )

    loss = torch.nn.functional.mse_loss(
        predicted_noise,
        target_noise,
    )

    loss.backward()

    assert torch.isfinite(
        loss
    )

    parameters_with_gradient = 0

    for parameter in model.parameters():
        if parameter.grad is not None:
            parameters_with_gradient += 1

            assert torch.isfinite(
                parameter.grad
            ).all()

    assert parameters_with_gradient > 0


def test_unet_parameter_count():
    model = UNet()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"UNet parameters: "
        f"{parameter_count:,}"
    )

    assert parameter_count > 0