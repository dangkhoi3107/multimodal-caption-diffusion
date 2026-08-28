from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.product_dataset import ProductImageDataset
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.trainer import training_step
from src.diffusion.unet import UNet


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

CHECKPOINT_PATH = Path(
    "checkpoints/phase1_unconditional/best.pt"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # -------------------------
    # Dataset
    # -------------------------

    dataset = ProductImageDataset(
        Path(
            config["data"]["train_metadata"]
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    x_0 = batch["image"].to(
        device=device,
        dtype=torch.float32,
    )

    print(
        "Resume batch:",
        tuple(x_0.shape),
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler = DDPMScheduler(
        num_timesteps=int(
            config["diffusion"]["num_timesteps"]
        ),
        beta_start=float(
            config["diffusion"]["beta_start"]
        ),
        beta_end=float(
            config["diffusion"]["beta_end"]
        ),
    )

    # -------------------------
    # Model + optimizer
    # -------------------------

    model = UNet(
        in_channels=int(
            config["model"]["in_channels"]
        ),
        out_channels=int(
            config["model"]["out_channels"]
        ),
        base_channels=int(
            config["model"]["base_channels"]
        ),
        time_embedding_dim=int(
            config["model"]["time_embedding_dim"]
        ),
        time_dim=int(
            config["model"]["time_dim"]
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"]["learning_rate"]
        ),
    )

    # -------------------------
    # Load checkpoint
    # -------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print(
        "Loaded epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint train loss:",
        checkpoint["train_loss"],
    )

    print(
        "Checkpoint valid loss:",
        checkpoint["valid_loss"],
    )

    # -------------------------
    # Snapshot parameters
    # -------------------------

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    # -------------------------
    # Exactly one resumed step
    # -------------------------

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(9999)

    metrics = training_step(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        optimizer=optimizer,
        generator=generator,
    )

    # -------------------------
    # Verify parameter update
    # -------------------------

    changed = False

    for name, parameter in model.named_parameters():
        if not torch.equal(
            before[name],
            parameter.detach(),
        ):
            changed = True
            break

    loss = float(
        metrics["loss"]
    )

    if not torch.isfinite(
        torch.tensor(loss)
    ):
        raise RuntimeError(
            "Resume loss is NaN/Inf"
        )

    if not changed:
        raise RuntimeError(
            "Parameters did not update"
        )

    print()
    print(
        "Resume step loss:",
        loss,
    )

    print(
        "Parameters updated:",
        changed,
    )

    print()
    print(
        "PASS: checkpoint can resume training."
    )


if __name__ == "__main__":
    main()