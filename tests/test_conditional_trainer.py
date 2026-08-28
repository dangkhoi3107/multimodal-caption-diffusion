import torch
from torch import nn

from src.diffusion.conditional_trainer import (
    compute_conditional_diffusion_loss,
    conditional_training_step,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


class TinyConditionalModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.null_class_id = (
            num_classes
        )

        self.embedding = nn.Embedding(
            num_classes + 1,
            3,
        )

        self.last_class_ids = None

    def forward(
        self,
        x_t,
        timesteps,
        class_ids,
    ):
        self.last_class_ids = (
            class_ids
            .detach()
            .clone()
        )

        condition = self.embedding(
            class_ids
        )

        condition = condition[
            :,
            :,
            None,
            None,
        ]

        return (
            torch.zeros_like(
                x_t
            )
            + condition
        )


def make_scheduler():
    return DDPMScheduler(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )


def test_conditional_loss_is_scalar_and_finite():
    model = TinyConditionalModel()

    scheduler = make_scheduler()

    x_0 = torch.randn(
        4,
        3,
        16,
        16,
    )

    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x_0
    )

    class_ids = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    loss = compute_conditional_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
        class_ids=class_ids,
    )

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )


def test_dropout_one_uses_only_null_class():
    model = TinyConditionalModel(
        num_classes=3,
    )

    scheduler = make_scheduler()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x_0 = torch.randn(
        4,
        3,
        16,
        16,
    )

    class_ids = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    metrics = conditional_training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        class_ids=class_ids,
        optimizer=optimizer,
        condition_dropout=1.0,
        null_class_id=3,
    )

    expected = torch.tensor(
        [3, 3, 3, 3],
        dtype=torch.long,
    )

    assert torch.equal(
        model.last_class_ids,
        expected,
    )

    assert (
        metrics["dropped_count"]
        == 4
    )

    assert (
        metrics["dropped_fraction"]
        == 1.0
    )


def test_dropout_zero_keeps_real_classes():
    model = TinyConditionalModel(
        num_classes=3,
    )

    scheduler = make_scheduler()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x_0 = torch.randn(
        3,
        3,
        16,
        16,
    )

    class_ids = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    metrics = conditional_training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        class_ids=class_ids,
        optimizer=optimizer,
        condition_dropout=0.0,
        null_class_id=3,
    )

    assert torch.equal(
        model.last_class_ids,
        class_ids,
    )

    assert (
        metrics["dropped_count"]
        == 0
    )


def test_training_step_updates_parameters():
    model = TinyConditionalModel(
        num_classes=3,
    )

    scheduler = make_scheduler()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    x_0 = torch.randn(
        4,
        3,
        16,
        16,
    )

    class_ids = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    before = (
        model.embedding.weight
        .detach()
        .clone()
    )

    metrics = conditional_training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        class_ids=class_ids,
        optimizer=optimizer,
        condition_dropout=0.0,
        null_class_id=3,
    )

    after = (
        model.embedding.weight
        .detach()
    )

    assert torch.isfinite(
        torch.tensor(
            metrics["loss"]
        )
    )

    assert not torch.equal(
        before,
        after,
    )


def test_training_step_with_seed_is_reproducible():
    torch.manual_seed(42)

    model_a = TinyConditionalModel()
    model_b = TinyConditionalModel()

    model_b.load_state_dict(
        model_a.state_dict()
    )

    scheduler_a = make_scheduler()
    scheduler_b = make_scheduler()

    optimizer_a = torch.optim.SGD(
        model_a.parameters(),
        lr=0.0,
    )

    optimizer_b = torch.optim.SGD(
        model_b.parameters(),
        lr=0.0,
    )

    x_0 = torch.randn(
        8,
        3,
        16,
        16,
    )

    class_ids = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1],
        dtype=torch.long,
    )

    generator_a = (
        torch.Generator()
    )

    generator_b = (
        torch.Generator()
    )

    generator_a.manual_seed(123)
    generator_b.manual_seed(123)

    metrics_a = conditional_training_step(
        model=model_a,
        scheduler=scheduler_a,
        x_0=x_0,
        class_ids=class_ids,
        optimizer=optimizer_a,
        condition_dropout=0.5,
        null_class_id=3,
        generator=generator_a,
    )

    metrics_b = conditional_training_step(
        model=model_b,
        scheduler=scheduler_b,
        x_0=x_0,
        class_ids=class_ids,
        optimizer=optimizer_b,
        condition_dropout=0.5,
        null_class_id=3,
        generator=generator_b,
    )

    assert torch.equal(
        model_a.last_class_ids,
        model_b.last_class_ids,
    )

    assert (
        metrics_a["loss"]
        == metrics_b["loss"]
    )

    assert (
        metrics_a[
            "dropped_count"
        ]
        == metrics_b[
            "dropped_count"
        ]
    )