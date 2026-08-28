import torch

from src.diffusion.scheduler import (
    DDPMScheduler,
    extract,
)


def test_extract_for_image_batch():
    values = torch.arange(
        10,
        dtype=torch.float32,
    )

    timesteps = torch.tensor(
        [0, 3, 9],
        dtype=torch.long,
    )

    result = extract(
        values=values,
        timesteps=timesteps,
        target_shape=(3, 3, 64, 64),
    )

    assert result.shape == (
        3,
        1,
        1,
        1,
    )

    assert torch.allclose(
        result[:, 0, 0, 0],
        torch.tensor(
            [0.0, 3.0, 9.0]
        ),
    )


def test_ddpm_q_sample_shape():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    x_0 = torch.randn(
        4,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x_0
    )

    x_t = scheduler.q_sample(
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    assert x_t.shape == (
        4,
        3,
        64,
        64,
    )

    assert torch.isfinite(
        x_t
    ).all()


def test_ddpm_q_sample_matches_formula():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    x_0 = torch.randn(
        4,
        3,
        8,
        8,
    )

    noise = torch.randn_like(
        x_0
    )

    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
    )

    x_t = scheduler.q_sample(
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    alpha_bar_t = (
        scheduler.alpha_bars[
            timesteps
        ]
        .reshape(4, 1, 1, 1)
    )

    expected = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(
            1.0 - alpha_bar_t
        ) * noise
    )

    assert torch.allclose(
        x_t,
        expected,
    )