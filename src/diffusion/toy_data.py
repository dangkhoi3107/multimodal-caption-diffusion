import torch


def sample_gaussian_mixture(
    num_samples: int,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)

    mean_1 = torch.tensor([-2.0, 0.0])
    mean_2 = torch.tensor([2.0, 0.0])

    covariance = torch.tensor([
        [0.5, 0.0],
        [0.0, 0.5], # Phương sai = 0.5
    ]) 

    # tạo distribution
    distribution_1 = torch.distributions.MultivariateNormal(
        loc=mean_1,
        covariance_matrix=covariance,
    )

    distribution_2 = torch.distributions.MultivariateNormal(
        loc=mean_2,
        covariance_matrix=covariance,
    )

    num_samples_1 = num_samples // 2
    num_samples_2 = num_samples - num_samples_1

    samples_1 = distribution_1.sample((num_samples_1,))
    samples_2 = distribution_2.sample((num_samples_2,))

    samples = torch.cat(
        [samples_1, samples_2],
        dim=0,
    )

    return samples