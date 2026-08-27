import torch 

def test_gaussian_2d_statistics():
    torch.manual_seed(36)

    mean = torch.tensor([2.0, -1.0]) 

    # variance(std^2): Đo lường độ phân tán so với mean
    # std: độ lệch chuẩn: đo lường độ phân tán. std càng lớn, data càng giao động
    # covariance: Đo lường mức độ biến thiên. Trong diffusion thường ép cov =0 để thêm nhiễu độc lập
    covariance = torch.tensor([
        [1.0, 0.5],
        [0.5, 2.0],
    ])

    # Hàm tạo phân phối nhiều chiều
    distribution = torch.distributions.MultivariateNormal(
        loc=mean, # location
        covariance_matrix=covariance,
    )

    samples = distribution.sample((100_000,))

    assert samples.shape == (100_000, 2)

    empirical_mean = samples.mean(dim=0)

    centered = samples - empirical_mean
    empirical_covariance = centered.T @ centered / (samples.shape[0] - 1)

    assert torch.allclose(
        empirical_mean,
        mean,
        atol=0.03,
    )

    assert torch.allclose(
        empirical_covariance,
        covariance,
        atol=0.05,
    )