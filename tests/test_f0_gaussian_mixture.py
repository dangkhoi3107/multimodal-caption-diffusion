import torch

from src.diffusion.toy_data import sample_gaussian_mixture


def test_gaussian_mixture_shape():
    samples = sample_gaussian_mixture(
        num_samples=10_000,
        seed=42,
    )

    assert samples.shape == (10_000, 2)
    assert torch.isfinite(samples).all()


def test_gaussian_mixture_mean():
    samples = sample_gaussian_mixture(
        num_samples=100_000,
        seed=42,
    )

    empirical_mean = samples.mean(dim=0)

    expected_mean = torch.tensor([0.0, 0.0])

    assert torch.allclose(
        empirical_mean,
        expected_mean,
        atol=0.05,
    )