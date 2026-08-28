import torch
from src.diffusion.scheduler import (
    compute_diffusion_coefficients,
    linear_beta_schedule,
    q_sample,
)

from src.diffusion.scheduler import (
    compute_diffusion_coefficients,
    linear_beta_schedule,
)
from src.diffusion.scheduler import (
    compute_diffusion_coefficients,
    linear_beta_schedule,
    q_sample,
    q_sample_iterative,
)
from src.diffusion.toy_data import sample_gaussian_mixture


def test_linear_beta_schedule_shape():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    assert betas.shape == (1000,)


def test_linear_beta_schedule_range():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    assert torch.all(betas > 0)
    assert torch.all(betas < 1)


def test_linear_beta_schedule_is_increasing():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    assert torch.all(
        betas[1:] > betas[:-1]
    )


def test_diffusion_coefficients():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    alphas, alpha_bars = compute_diffusion_coefficients(
        betas
    )

    assert alphas.shape == (1000,)
    assert alpha_bars.shape == (1000,)

    assert torch.allclose(
        alphas,
        1.0 - betas,
    )

    assert torch.all(
        alpha_bars[1:] < alpha_bars[:-1]
    )

    assert alpha_bars[0] > alpha_bars[-1]


def test_q_sample_shape():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    _, alpha_bars = compute_diffusion_coefficients(
        betas
    )

    x_0 = torch.randn(100, 2)
    noise = torch.randn_like(x_0)

    x_t = q_sample(
        x_0=x_0,
        t=500,
        alpha_bars=alpha_bars,
        noise=noise,
    )

    assert x_t.shape == x_0.shape


def test_q_sample_is_finite():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    _, alpha_bars = compute_diffusion_coefficients(
        betas
    )

    x_0 = torch.randn(100, 2)

    x_t = q_sample(
        x_0=x_0,
        t=500,
        alpha_bars=alpha_bars,
    )

    assert torch.isfinite(x_t).all()


def test_q_sample_matches_formula():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    _, alpha_bars = compute_diffusion_coefficients(
        betas
    )

    x_0 = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    noise = torch.tensor([
        [0.5, -0.5],
        [1.0, -1.0],
    ])

    t = 100

    x_t = q_sample(
        x_0=x_0,
        t=t,
        alpha_bars=alpha_bars,
        noise=noise,
    )

    alpha_bar_t = alpha_bars[t]

    expected = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    assert torch.allclose(
        x_t,
        expected,
    )


def test_q_sample_iterative_shape():
    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    alphas, _ = compute_diffusion_coefficients(
        betas
    )

    x_0 = torch.randn(100, 2)

    x_t = q_sample_iterative(
        x_0=x_0,
        t=100,
        alphas=alphas,
    )

    assert x_t.shape == x_0.shape
    assert torch.isfinite(x_t).all()


def test_closed_form_matches_iterative_distribution():
    torch.manual_seed(42)

    num_samples = 20_000
    t = 100

    x_0 = sample_gaussian_mixture(
        num_samples=num_samples,
        seed=42,
    )

    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    alphas, alpha_bars = compute_diffusion_coefficients(
        betas
    )

    # Iterative:
    # x0 -> x1 -> ... -> xt
    torch.manual_seed(123)

    x_t_iterative = q_sample_iterative(
        x_0=x_0,
        t=t,
        alphas=alphas,
    )

    # Closed-form:
    # x0 -> xt directly
    torch.manual_seed(456)

    noise = torch.randn_like(x_0)

    x_t_closed = q_sample(
        x_0=x_0,
        t=t,
        alpha_bars=alpha_bars,
        noise=noise,
    )

    iterative_mean = x_t_iterative.mean(dim=0)
    closed_mean = x_t_closed.mean(dim=0)

    iterative_centered = (
        x_t_iterative - iterative_mean
    )

    closed_centered = (
        x_t_closed - closed_mean
    )

    iterative_cov = (
        iterative_centered.T
        @ iterative_centered
        / (num_samples - 1)
    )

    closed_cov = (
        closed_centered.T
        @ closed_centered
        / (num_samples - 1)
    )

    assert torch.allclose(
        iterative_mean,
        closed_mean,
        atol=0.05,
    )

    assert torch.allclose(
        iterative_cov,
        closed_cov,
        atol=0.08,
    )

def test_posterior_variance_shape():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    assert (
        scheduler.posterior_variance.shape
        == (1000,)
    )

    assert torch.all(
        scheduler.posterior_variance >= 0
    )

    assert torch.allclose(
        scheduler.posterior_variance[0],
        torch.tensor(0.0),
    )


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

    mean, variance = (
        scheduler.p_mean_variance(
            model_output=predicted_noise,
            x_t=x_t,
            timesteps=timesteps,
        )
    )

    assert mean.shape == x_t.shape

    assert variance.shape == (
        4,
        1,
        1,
        1,
    )

    assert torch.isfinite(
        mean
    ).all()

    assert torch.isfinite(
        variance
    ).all()


def test_p_mean_matches_formula():
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

    mean, _ = (
        scheduler.p_mean_variance(
            model_output=predicted_noise,
            x_t=x_t,
            timesteps=timesteps,
        )
    )

    beta_t = (
        scheduler.betas[
            timesteps
        ].reshape(
            2, 1, 1, 1
        )
    )

    alpha_t = (
        scheduler.alphas[
            timesteps
        ].reshape(
            2, 1, 1, 1
        )
    )

    alpha_bar_t = (
        scheduler.alpha_bars[
            timesteps
        ].reshape(
            2, 1, 1, 1
        )
    )

    expected = (
        1.0
        / torch.sqrt(alpha_t)
    ) * (
        x_t
        - (
            beta_t
            / torch.sqrt(
                1.0 - alpha_bar_t
            )
        )
        * predicted_noise
    )

    assert torch.allclose(
        mean,
        expected,
    )


class ZeroModel(torch.nn.Module):
    def forward(
        self,
        x,
        timesteps,
    ):
        return torch.zeros_like(x)


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

    mean, _ = (
        scheduler.p_mean_variance(
            model_output=model(
                x_t,
                timesteps,
            ),
            x_t=x_t,
            timesteps=timesteps,
        )
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

    generator1 = torch.Generator()
    generator1.manual_seed(1)

    generator2 = torch.Generator()
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