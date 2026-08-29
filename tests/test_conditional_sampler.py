import torch
from torch import nn

from src.diffusion.conditional_sampler import (
    classifier_free_guidance,
    sample_ddpm_cfg,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


class TinyCFGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.null_class_id = 3

        self.scale = nn.Parameter(
            torch.tensor(0.01)
        )

    def forward(
        self,
        x_t,
        timesteps,
        class_ids,
    ):
        condition = (
            class_ids.float()
            .reshape(
                -1,
                1,
                1,
                1,
            )
        )

        return (
            torch.zeros_like(
                x_t
            )
            + self.scale
            * condition
        )


def make_scheduler():
    return DDPMScheduler(
        num_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
    )


def test_cfg_scale_zero_is_unconditional():
    epsilon_uncond = torch.randn(
        2,
        3,
        8,
        8,
    )

    epsilon_cond = torch.randn_like(
        epsilon_uncond
    )

    output = classifier_free_guidance(
        epsilon_uncond,
        epsilon_cond,
        guidance_scale=0.0,
    )

    assert torch.allclose(
        output,
        epsilon_uncond,
    )


def test_cfg_scale_one_is_conditional():
    epsilon_uncond = torch.randn(
        2,
        3,
        8,
        8,
    )

    epsilon_cond = torch.randn_like(
        epsilon_uncond
    )

    output = classifier_free_guidance(
        epsilon_uncond,
        epsilon_cond,
        guidance_scale=1.0,
    )

    torch.testing.assert_close(
        output,
        epsilon_cond,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cfg_formula_scale_three():
    epsilon_uncond = torch.ones(
        1,
        3,
        4,
        4,
    )

    epsilon_cond = torch.full_like(
        epsilon_uncond,
        2.0,
    )

    output = classifier_free_guidance(
        epsilon_uncond,
        epsilon_cond,
        guidance_scale=3.0,
    )

    expected = torch.full_like(
        epsilon_uncond,
        4.0,
    )

    assert torch.allclose(
        output,
        expected,
    )


def test_cfg_sampler_shape_and_finite():
    device = torch.device(
        "cpu"
    )

    model = TinyCFGModel().to(
        device
    )

    scheduler = make_scheduler()

    class_ids = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        42
    )

    samples = sample_ddpm_cfg(
        model=model,
        scheduler=scheduler,
        class_ids=class_ids,
        shape=(
            2,
            3,
            8,
            8,
        ),
        device=device,
        guidance_scale=3.0,
        generator=generator,
    )

    assert samples.shape == (
        2,
        3,
        8,
        8,
    )

    assert torch.isfinite(
        samples
    ).all()


def test_cfg_sampler_is_deterministic():
    device = torch.device(
        "cpu"
    )

    model = TinyCFGModel().to(
        device
    )

    scheduler = make_scheduler()

    class_ids = torch.tensor(
        [1],
        dtype=torch.long,
    )

    generator_a = torch.Generator(
        device="cpu"
    )

    generator_b = torch.Generator(
        device="cpu"
    )

    generator_a.manual_seed(
        123
    )

    generator_b.manual_seed(
        123
    )

    sample_a = sample_ddpm_cfg(
        model=model,
        scheduler=scheduler,
        class_ids=class_ids,
        shape=(
            1,
            3,
            8,
            8,
        ),
        device=device,
        guidance_scale=2.0,
        generator=generator_a,
    )

    sample_b = sample_ddpm_cfg(
        model=model,
        scheduler=scheduler,
        class_ids=class_ids,
        shape=(
            1,
            3,
            8,
            8,
        ),
        device=device,
        guidance_scale=2.0,
        generator=generator_b,
    )

    assert torch.allclose(
        sample_a,
        sample_b,
    )