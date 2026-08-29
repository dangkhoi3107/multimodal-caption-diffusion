import torch
from torch import nn

from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.text_sampler import (
    sample_ddpm_text_cfg,
)


class TinyTextModel(nn.Module):
    def forward(
        self,
        x_t,
        timesteps,
        token_ids,
        padding_mask,
    ):
        text_value = (
            token_ids.float()
            * padding_mask.float()
        ).sum(
            dim=1
        )

        text_value = (
            text_value
            / 100.0
        ).reshape(
            -1,
            1,
            1,
            1,
        )

        return (
            0.1 * x_t
            + text_value
        )


def make_scheduler():
    return DDPMScheduler(
        num_timesteps=5,
        beta_start=0.0001,
        beta_end=0.02,
    )


def make_prompt_a():
    ids = torch.tensor(
        [[
            1, 4, 15, 12, 10,
            14, 2, 0, 0, 0,
        ]],
        dtype=torch.long,
    )

    return (
        ids,
        ids != 0,
    )


def make_prompt_b():
    ids = torch.tensor(
        [[
            1, 4, 5, 9, 8,
            17, 2, 0, 0, 0,
        ]],
        dtype=torch.long,
    )

    return (
        ids,
        ids != 0,
    )


def test_text_cfg_output_shape():
    model = TinyTextModel()
    scheduler = make_scheduler()

    ids, mask = make_prompt_a()

    generator = (
        torch.Generator()
        .manual_seed(42)
    )

    sample = sample_ddpm_text_cfg(
        model=model,
        scheduler=scheduler,
        token_ids=ids,
        padding_mask=mask,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        guidance_scale=1.0,
        generator=generator,
    )

    assert sample.shape == (
        1,
        3,
        8,
        8,
    )

    assert torch.isfinite(
        sample
    ).all()


def test_cfg_zero_ignores_prompt():
    model = TinyTextModel()
    scheduler = make_scheduler()

    ids_a, mask_a = make_prompt_a()
    ids_b, mask_b = make_prompt_b()

    generator_a = (
        torch.Generator()
        .manual_seed(123)
    )

    generator_b = (
        torch.Generator()
        .manual_seed(123)
    )

    sample_a = sample_ddpm_text_cfg(
        model=model,
        scheduler=scheduler,
        token_ids=ids_a,
        padding_mask=mask_a,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        guidance_scale=0.0,
        generator=generator_a,
    )

    sample_b = sample_ddpm_text_cfg(
        model=model,
        scheduler=scheduler,
        token_ids=ids_b,
        padding_mask=mask_b,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        guidance_scale=0.0,
        generator=generator_b,
    )

    torch.testing.assert_close(
        sample_a,
        sample_b,
    )


def test_cfg_one_responds_to_prompt():
    model = TinyTextModel()
    scheduler = make_scheduler()

    ids_a, mask_a = make_prompt_a()
    ids_b, mask_b = make_prompt_b()

    generator_a = (
        torch.Generator()
        .manual_seed(123)
    )

    generator_b = (
        torch.Generator()
        .manual_seed(123)
    )

    sample_a = sample_ddpm_text_cfg(
        model=model,
        scheduler=scheduler,
        token_ids=ids_a,
        padding_mask=mask_a,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        guidance_scale=1.0,
        generator=generator_a,
    )

    sample_b = sample_ddpm_text_cfg(
        model=model,
        scheduler=scheduler,
        token_ids=ids_b,
        padding_mask=mask_b,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        shape=(1, 3, 8, 8),
        device=torch.device("cpu"),
        guidance_scale=1.0,
        generator=generator_b,
    )

    difference = (
        sample_a - sample_b
    ).abs().mean()

    assert (
        difference.item()
        > 0.0
    )


def test_negative_guidance_rejected():
    model = TinyTextModel()
    scheduler = make_scheduler()

    ids, mask = make_prompt_a()

    try:
        sample_ddpm_text_cfg(
            model=model,
            scheduler=scheduler,
            token_ids=ids,
            padding_mask=mask,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            shape=(1, 3, 8, 8),
            device=torch.device("cpu"),
            guidance_scale=-1.0,
        )
    except ValueError:
        return

    raise AssertionError(
        "negative guidance should fail"
    )