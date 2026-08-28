import torch
from torch import nn

from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.trainer import (
    compute_diffusion_loss,
    train_epoch,
    training_step,
    validate_epoch,
)
from torch.utils.data import DataLoader


class TinyNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            3,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x_t,
        timesteps,
    ):
        return self.conv(x_t)


def test_diffusion_loss_is_scalar():
    model = TinyNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

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

    loss = compute_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diffusion_loss_backward():
    model = TinyNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    x_0 = torch.randn(
        2,
        3,
        16,
        16,
    )

    timesteps = torch.tensor(
        [100, 700],
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x_0
    )

    loss = compute_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None

        assert torch.isfinite(
            parameter.grad
        ).all()


class ZeroNoisePredictor(nn.Module):
    def forward(
        self,
        x_t,
        timesteps,
    ):
        return torch.zeros_like(x_t)


def test_diffusion_loss_matches_expected_mse():
    model = ZeroNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=100
    )

    x_0 = torch.randn(
        2,
        3,
        8,
        8,
    )

    timesteps = torch.tensor(
        [10, 50],
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x_0
    )

    loss = compute_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        timesteps=timesteps,
        noise=noise,
    )

    expected = (
        noise.square().mean()
    )

    assert torch.allclose(
        loss,
        expected,
    )



def test_training_step_updates_parameters():
    torch.manual_seed(42)

    model = TinyNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=100
    )

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

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        optimizer=optimizer,
    )

    parameters_after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert metrics["loss"] > 0

    assert (
        0
        <= metrics["timestep_min"]
        < 100
    )

    assert (
        0
        <= metrics["timestep_max"]
        < 100
    )

    changed = [
        not torch.allclose(
            before,
            after,
        )
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    ]

    assert any(changed)


def test_training_step_batch_size():
    model = TinyNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x_0 = torch.randn(
        8,
        3,
        16,
        16,
    )

    metrics = training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        optimizer=optimizer,
    )

    assert metrics["batch_size"] == 8

    assert (
        0
        <= metrics["timestep_min"]
        <= metrics["timestep_max"]
        < 1000
    )

class TinyImageDataset:
    def __init__(
        self,
        num_images: int,
    ):
        self.images = torch.randn(
            num_images,
            3,
            16,
            16,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(
        self,
        index,
    ):
        return {
            "image": self.images[index]
        }


def test_train_epoch():
    model = TinyNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=100
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    dataset = TinyImageDataset(
        num_images=10
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    metrics = train_epoch(
        model=model,
        scheduler=scheduler,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert metrics["num_samples"] == 10

    assert metrics["loss"] > 0

    assert torch.isfinite(
        torch.tensor(
            metrics["loss"]
        )
    )


def test_validate_epoch_does_not_update_parameters():
    model = TinyNoisePredictor()

    scheduler = DDPMScheduler(
        num_timesteps=100
    )

    dataset = TinyImageDataset(
        num_images=8
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = validate_epoch(
        model=model,
        scheduler=scheduler,
        dataloader=loader,
        device=torch.device("cpu"),
    )

    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert metrics["num_samples"] == 8
    assert metrics["loss"] > 0

    for parameter_before, parameter_after in zip(
        before,
        after,
    ):
        assert torch.equal(
            parameter_before,
            parameter_after,
        )