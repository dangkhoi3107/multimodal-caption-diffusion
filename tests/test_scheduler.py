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