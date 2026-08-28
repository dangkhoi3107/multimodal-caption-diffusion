import torch
from torch import nn

from src.diffusion.sampler import sample_ddpm
from src.diffusion.scheduler import DDPMScheduler


class ZeroNoiseModel(nn.Module):
    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x_t)


def test_sample_ddpm_shape():
    model = ZeroNoiseModel()

    scheduler = DDPMScheduler(
        num_timesteps=10,
        beta_start=1e-4,
        beta_end=0.02,
    )

    sample = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(2, 3, 16, 16),
        device=torch.device("cpu"),
    )

    assert sample.shape == (
        2,
        3,
        16,
        16,
    )

    assert torch.isfinite(
        sample
    ).all()


def test_sample_ddpm_is_deterministic_with_generator():
    model = ZeroNoiseModel()

    scheduler = DDPMScheduler(
        num_timesteps=10
    )

    generator1 = torch.Generator(
        device="cpu"
    )
    generator1.manual_seed(42)

    sample1 = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        generator=generator1,
    )

    generator2 = torch.Generator(
        device="cpu"
    )
    generator2.manual_seed(42)

    sample2 = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        generator=generator2,
    )

    assert torch.allclose(
        sample1,
        sample2,
    )


def test_sample_ddpm_different_seeds():
    model = ZeroNoiseModel()

    scheduler = DDPMScheduler(
        num_timesteps=10
    )

    generator1 = torch.Generator(
        device="cpu"
    )
    generator1.manual_seed(1)

    generator2 = torch.Generator(
        device="cpu"
    )
    generator2.manual_seed(2)

    sample1 = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        generator=generator1,
    )

    sample2 = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        generator=generator2,
    )

    assert not torch.allclose(
        sample1,
        sample2,
    )


def test_sample_ddpm_returns_trajectory():
    model = ZeroNoiseModel()

    scheduler = DDPMScheduler(
        num_timesteps=10
    )

    sample, trajectory = sample_ddpm(
        model=model,
        scheduler=scheduler,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        return_trajectory=True,
        trajectory_interval=2,
    )

    assert sample.shape == (
        1,
        3,
        8,
        8,
    )

    assert len(trajectory) > 1

    for state in trajectory:
        assert state.shape == (
            1,
            3,
            8,
            8,
        )

        assert torch.isfinite(
            state
        ).all()
