import torch

from src.diffusion.toy_score import (
    finite_difference_score,
    gaussian_score,
)


def test_gaussian_score_identity_covariance():
    mean = torch.tensor([0.0, 0.0])

    covariance = torch.eye(2)

    x = torch.tensor([
        [3.0, 2.0],
        [-1.0, 4.0],
        [0.0, 0.0],
    ])

    score = gaussian_score(
        x=x,
        mean=mean,
        covariance=covariance,
    )

    expected = -x

    assert score.shape == x.shape

    assert torch.allclose(
        score,
        expected,
    )


def test_gaussian_score_at_mean_is_zero():
    mean = torch.tensor([2.0, -1.0])

    covariance = torch.tensor([
        [1.0, 0.5],
        [0.5, 2.0],
    ])

    x = mean.unsqueeze(0)

    score = gaussian_score(
        x=x,
        mean=mean,
        covariance=covariance,
    )

    expected = torch.zeros_like(x)

    assert torch.allclose(
        score,
        expected,
        atol=1e-6,
    )


def test_gaussian_score_is_finite():
    mean = torch.tensor([0.0, 0.0])

    covariance = torch.tensor([
        [1.0, 0.3],
        [0.3, 2.0],
    ])

    x = torch.randn(100, 2)

    score = gaussian_score(
        x=x,
        mean=mean,
        covariance=covariance,
    )

    assert score.shape == (100, 2)
    assert torch.isfinite(score).all()


def test_gaussian_score_matches_finite_difference():
    mean = torch.tensor([0.5, -1.0])

    covariance = torch.tensor([
        [1.5, 0.3],
        [0.3, 0.8],
    ])

    x = torch.tensor([
        [1.0, 0.5],
        [-1.0, 2.0],
        [2.0, -2.0],
    ])

    analytic_score = gaussian_score(
        x=x,
        mean=mean,
        covariance=covariance,
    )

    numerical_score = finite_difference_score(
        x=x,
        mean=mean,
        covariance=covariance,
        h=1e-3,
    )

    assert torch.allclose(
        analytic_score,
        numerical_score,
        atol=1e-3,
        rtol=1e-3,
    )