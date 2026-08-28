import torch

from src.diffusion.scheduler import (
    DDPMScheduler,
    extract,
)


class ZeroModel(torch.nn.Module):
    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x_t)


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
        torch.sqrt(
            alpha_bar_t
        )
        * x_0
        + torch.sqrt(
            1.0 - alpha_bar_t
        )
        * noise
    )

    assert torch.allclose(
        x_t,
        expected,
    )


def test_posterior_variance_and_coefficients():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    assert scheduler.posterior_variance.shape == (
        1000,
    )

    assert scheduler.posterior_mean_coef1.shape == (
        1000,
    )

    assert scheduler.posterior_mean_coef2.shape == (
        1000,
    )

    assert torch.all(
        scheduler.posterior_variance >= 0
    )

    assert torch.allclose(
        scheduler.posterior_variance[0],
        torch.tensor(0.0),
    )

    assert torch.isfinite(
        scheduler.posterior_mean_coef1
    ).all()

    assert torch.isfinite(
        scheduler.posterior_mean_coef2
    ).all()


def test_p_mean_variance_shapes():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    x_t = torch.randn(
        4,
        3,
        16,
        16,
    )

    predicted_noise = torch.randn_like(
        x_t
    )

    timesteps = torch.tensor(
        [1, 100, 500, 999],
        dtype=torch.long,
    )

    (
        mean,
        variance,
        predicted_x_0,
    ) = scheduler.p_mean_variance(
        model_output=predicted_noise,
        x_t=x_t,
        timesteps=timesteps,
    )

    assert mean.shape == x_t.shape

    assert variance.shape == (
        4,
        1,
        1,
        1,
    )

    assert predicted_x_0.shape == x_t.shape

    assert torch.isfinite(
        mean
    ).all()

    assert torch.isfinite(
        variance
    ).all()

    assert torch.isfinite(
        predicted_x_0
    ).all()


def test_p_mean_variance_clips_predicted_x0():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    ) * 10.0

    predicted_noise = torch.randn_like(
        x_t
    )

    timesteps = torch.tensor(
        [500, 999],
        dtype=torch.long,
    )

    (
        _mean,
        _variance,
        predicted_x_0,
    ) = scheduler.p_mean_variance(
        model_output=predicted_noise,
        x_t=x_t,
        timesteps=timesteps,
        clip_denoised=True,
    )

    assert predicted_x_0.min() >= -1.0
    assert predicted_x_0.max() <= 1.0


def test_p_mean_matches_epsilon_formula_without_clipping():
    scheduler = DDPMScheduler(
        num_timesteps=100
    )

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    )

    predicted_noise = torch.randn_like(
        x_t
    )

    timesteps = torch.tensor(
        [10, 50],
        dtype=torch.long,
    )

    (
        mean,
        _variance,
        _predicted_x_0,
    ) = scheduler.p_mean_variance(
        model_output=predicted_noise,
        x_t=x_t,
        timesteps=timesteps,
        clip_denoised=False,
    )

    beta_t = (
        scheduler.betas[
            timesteps
        ]
        .reshape(
            2,
            1,
            1,
            1,
        )
    )

    alpha_t = (
        scheduler.alphas[
            timesteps
        ]
        .reshape(
            2,
            1,
            1,
            1,
        )
    )

    alpha_bar_t = (
        scheduler.alpha_bars[
            timesteps
        ]
        .reshape(
            2,
            1,
            1,
            1,
        )
    )

    expected = (
        1.0
        / torch.sqrt(
            alpha_t
        )
    ) * (
        x_t
        - (
            beta_t
            / torch.sqrt(
                1.0
                - alpha_bar_t
            )
        )
        * predicted_noise
    )

    assert torch.allclose(
        mean,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_p_sample_t_zero_has_no_random_noise():
    scheduler = DDPMScheduler(
        num_timesteps=100
    )

    model = ZeroModel()

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.zeros(
        2,
        dtype=torch.long,
    )

    model_output = model(
        x_t,
        timesteps,
    )

    (
        mean,
        _variance,
        _predicted_x_0,
    ) = scheduler.p_mean_variance(
        model_output=model_output,
        x_t=x_t,
        timesteps=timesteps,
        clip_denoised=True,
    )

    sample = scheduler.p_sample(
        model=model,
        x_t=x_t,
        timesteps=timesteps,
    )

    assert torch.allclose(
        sample,
        mean,
    )


def test_p_sample_t_positive_is_stochastic():
    scheduler = DDPMScheduler(
        num_timesteps=100
    )

    model = ZeroModel()

    x_t = torch.randn(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.tensor(
        [50, 50],
        dtype=torch.long,
    )

    generator1 = torch.Generator(
        device="cpu"
    )
    generator1.manual_seed(1)

    generator2 = torch.Generator(
        device="cpu"
    )
    generator2.manual_seed(2)

    sample1 = scheduler.p_sample(
        model=model,
        x_t=x_t,
        timesteps=timesteps,
        generator=generator1,
    )

    sample2 = scheduler.p_sample(
        model=model,
        x_t=x_t,
        timesteps=timesteps,
        generator=generator2,
    )

    assert not torch.allclose(
        sample1,
        sample2,
    )
