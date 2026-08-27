"""
    grouth truth cho model
"""
import torch


def gaussian_score(
    x: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must have shape [N, D]")

    if mean.ndim != 1:
        raise ValueError("mean must have shape [D]")

    if covariance.ndim != 2:
        raise ValueError("covariance must have shape [D, D]")

    if x.shape[1] != mean.shape[0]:
        raise ValueError("x and mean dimensions do not match")

    dimension = mean.shape[0]

    if covariance.shape != (dimension, dimension):
        raise ValueError(
            "covariance shape must match the data dimension"
        )

    centered = x - mean

    score = -torch.linalg.solve(
        covariance,
        centered.T,
    ).T

    return score


def gaussian_log_density(
    x: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    distribution = torch.distributions.MultivariateNormal(
        loc=mean,
        covariance_matrix=covariance,
    )

    return distribution.log_prob(x)


def finite_difference_score(
    x: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    h: float = 1e-4,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must have shape [N, D]")

    num_samples, dimension = x.shape

    score = torch.zeros_like(x)

    for dim in range(dimension):
        offset = torch.zeros_like(x)

        offset[:, dim] = h

        log_p_plus = gaussian_log_density(
            x + offset,
            mean,
            covariance,
        )

        log_p_minus = gaussian_log_density(
            x - offset,
            mean,
            covariance,
        )

        score[:, dim] = (
            log_p_plus - log_p_minus
        ) / (2.0 * h)

    return score