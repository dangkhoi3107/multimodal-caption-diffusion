import torch
from torch import nn

from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.text_conditioning import (
    drop_text_condition,
)
from src.diffusion.text_trainer import (
    compute_text_diffusion_loss,
    text_training_step,
)


BOS_ID = 1
EOS_ID = 2
PAD_ID = 0


def make_tokens():
    token_ids = torch.tensor(
        [
            [
                1, 4, 15, 12, 10,
                14, 2, 0, 0, 0,
            ],
            [
                1, 4, 5, 9, 8,
                17, 2, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    return (
        token_ids,
        token_ids != PAD_ID,
    )


class TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(
                0.1
            )
        )

        self.last_token_ids = None
        self.last_padding_mask = None

    def forward(
        self,
        x_t,
        timesteps,
        token_ids,
        padding_mask,
    ):
        self.last_token_ids = (
            token_ids
            .detach()
            .clone()
        )

        self.last_padding_mask = (
            padding_mask
            .detach()
            .clone()
        )

        text_value = (
            token_ids.float()
            * padding_mask.float()
        ).sum(
            dim=1
        )

        text_value = (
            text_value
            / 100.0
        ).view(
            -1,
            1,
            1,
            1,
        )

        return (
            self.scale
            * x_t
            + text_value
        )


def make_scheduler():
    return DDPMScheduler(
        num_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
    )


def test_prompt_dropout_zero():
    token_ids, mask = (
        make_tokens()
    )

    (
        output_ids,
        output_mask,
        dropped,
    ) = drop_text_condition(
        token_ids=token_ids,
        padding_mask=mask,
        probability=0.0,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )

    assert torch.equal(
        output_ids,
        token_ids,
    )

    assert torch.equal(
        output_mask,
        mask,
    )

    assert not dropped.any()


def test_prompt_dropout_one():
    token_ids, mask = (
        make_tokens()
    )

    (
        output_ids,
        output_mask,
        dropped,
    ) = drop_text_condition(
        token_ids=token_ids,
        padding_mask=mask,
        probability=1.0,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )

    expected = torch.tensor(
        [
            [
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            [
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        output_ids,
        expected,
    )

    assert torch.equal(
        output_mask,
        expected != PAD_ID,
    )

    assert dropped.all()


def test_prompt_dropout_deterministic():
    token_ids, mask = (
        make_tokens()
    )

    generator_a = (
        torch.Generator()
        .manual_seed(42)
    )

    generator_b = (
        torch.Generator()
        .manual_seed(42)
    )

    result_a = (
        drop_text_condition(
            token_ids=token_ids,
            padding_mask=mask,
            probability=0.5,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            generator=generator_a,
        )
    )

    result_b = (
        drop_text_condition(
            token_ids=token_ids,
            padding_mask=mask,
            probability=0.5,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            generator=generator_b,
        )
    )

    for a, b in zip(
        result_a,
        result_b,
    ):
        assert torch.equal(
            a,
            b,
        )


def test_text_loss_scalar_finite():
    model = TinyTextModel()
    scheduler = make_scheduler()

    x_0 = torch.randn(
        2,
        3,
        16,
        16,
    )

    noise = torch.randn_like(
        x_0
    )

    timesteps = torch.tensor(
        [2, 7],
        dtype=torch.long,
    )

    token_ids, mask = (
        make_tokens()
    )

    loss = (
        compute_text_diffusion_loss(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
            token_ids=token_ids,
            padding_mask=mask,
        )
    )

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )


def test_training_step_full_dropout():
    model = TinyTextModel()

    scheduler = make_scheduler()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x_0 = torch.randn(
        2,
        3,
        16,
        16,
    )

    token_ids, mask = (
        make_tokens()
    )

    metrics = text_training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        token_ids=token_ids,
        padding_mask=mask,
        optimizer=optimizer,
        prompt_dropout=1.0,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )

    assert (
        metrics[
            "dropped_fraction"
        ]
        == 1.0
    )

    expected = torch.tensor(
        [
            [
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            [
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        model.last_token_ids,
        expected,
    )


def test_training_step_updates_parameter():
    model = TinyTextModel()

    scheduler = make_scheduler()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    before = (
        model.scale
        .detach()
        .clone()
    )

    x_0 = torch.randn(
        2,
        3,
        16,
        16,
    )

    token_ids, mask = (
        make_tokens()
    )

    text_training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        token_ids=token_ids,
        padding_mask=mask,
        optimizer=optimizer,
        prompt_dropout=0.1,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )

    after = (
        model.scale
        .detach()
        .clone()
    )

    assert not torch.equal(
        before,
        after,
    )