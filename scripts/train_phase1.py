import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.product_dataset import (
    ProductImageDataset,
)
from src.diffusion.sampler import (
    sample_ddpm,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.trainer import (
    train_epoch,
    validate_epoch,
)
from src.diffusion.unet import (
    UNet,
)


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

CHECKPOINT_DIR = Path(
    "checkpoints/phase1_unconditional"
)

OUTPUT_DIR = Path(
    "outputs/phase1_unconditional"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def tensor_to_image(
    tensor: torch.Tensor,
) -> np.ndarray:
    tensor = tensor.detach().cpu()

    # Chỉ clamp khi visualize.
    tensor = tensor.clamp(
        -1.0,
        1.0,
    )

    tensor = (
        tensor + 1.0
    ) / 2.0

    image = tensor.permute(
        1,
        2,
        0,
    ).numpy()

    return image


def save_sample_grid(
    samples: torch.Tensor,
    path: Path,
) -> None:
    num_images = samples.shape[0]

    fig, axes = plt.subplots(
        1,
        num_images,
        figsize=(
            num_images * 2.2,
            2.5,
        ),
    )

    if num_images == 1:
        axes = [axes]

    for index in range(
        num_images
    ):
        axes[index].imshow(
            tensor_to_image(
                samples[index]
            ),
            interpolation="nearest",
        )

        axes[index].set_title(
            f"sample {index}"
        )

        axes[index].axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def save_loss_curve(
    train_losses: list[float],
    valid_losses: list[float],
) -> None:
    epochs = range(
        1,
        len(train_losses) + 1,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        label="train",
    )

    plt.plot(
        epochs,
        valid_losses,
        label="valid",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "Phase 1 DDPM Training"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=150,
    )

    plt.close()


def save_checkpoint(
    path: Path,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    config: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "config": config,
        },
        path,
    )


def main() -> None:
    config = load_config()

    seed = int(
        config["training"]["seed"]
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = ProductImageDataset(
        Path(
            config["data"][
                "train_metadata"
            ]
        )
    )

    valid_dataset = ProductImageDataset(
        Path(
            config["data"][
                "valid_metadata"
            ]
        )
    )

    batch_size = int(
        config["training"][
            "batch_size"
        ]
    )

    num_workers = int(
        config["training"][
            "num_workers"
        ]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(
        "Train images:",
        len(train_dataset),
    )

    print(
        "Valid images:",
        len(valid_dataset),
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler = DDPMScheduler(
        num_timesteps=int(
            config["diffusion"][
                "num_timesteps"
            ]
        ),
        beta_start=float(
            config["diffusion"][
                "beta_start"
            ]
        ),
        beta_end=float(
            config["diffusion"][
                "beta_end"
            ]
        ),
    )

    # -------------------------
    # Model
    # -------------------------

    model = UNet(
        in_channels=int(
            config["model"][
                "in_channels"
            ]
        ),
        out_channels=int(
            config["model"][
                "out_channels"
            ]
        ),
        base_channels=int(
            config["model"][
                "base_channels"
            ]
        ),
        time_embedding_dim=int(
            config["model"][
                "time_embedding_dim"
            ]
        ),
        time_dim=int(
            config["model"][
                "time_dim"
            ]
        ),
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"][
                "learning_rate"
            ]
        ),
    )

    epochs = int(
        config["training"][
            "epochs"
        ]
    )

    checkpoint_interval = int(
        config["training"][
            "checkpoint_interval"
        ]
    )

    sample_interval = int(
        config["training"][
            "sample_interval"
        ]
    )

    num_sample_images = int(
        config["training"][
            "num_sample_images"
        ]
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Fixed generators
    # -------------------------

    training_noise_generator = (
        torch.Generator(
            device=device
        )
    )

    training_noise_generator.manual_seed(
        seed + 100
    )

    # Validation phải sử dụng cùng
    # random t/noise giữa các epoch
    # để valid loss so sánh công bằng.
    validation_seed = seed + 200

    sample_seed = seed + 300

    # -------------------------
    # Training
    # -------------------------

    train_losses = []
    valid_losses = []

    best_valid_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_metrics = train_epoch(
            model=model,
            scheduler=scheduler,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            generator=training_noise_generator,
        )

        # Reset generator mỗi epoch:
        # cùng validation t/noise.
        validation_generator = (
            torch.Generator(
                device=device
            )
        )

        validation_generator.manual_seed(
            validation_seed
        )

        valid_metrics = validate_epoch(
            model=model,
            scheduler=scheduler,
            dataloader=valid_loader,
            device=device,
            generator=validation_generator,
        )

        train_loss = float(
            train_metrics["loss"]
        )

        valid_loss = float(
            valid_metrics["loss"]
        )

        train_losses.append(
            train_loss
        )

        valid_losses.append(
            valid_loss
        )

        print(
            f"epoch={epoch:03d}/{epochs} "
            f"train={train_loss:.6f} "
            f"valid={valid_loss:.6f}"
        )

        # ---------------------
        # Best checkpoint
        # ---------------------

        if (
            valid_loss
            < best_valid_loss
        ):
            best_valid_loss = (
                valid_loss
            )

            save_checkpoint(
                path=(
                    CHECKPOINT_DIR
                    / "best.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                config=config,
            )

            print(
                "  saved best.pt"
            )

        # ---------------------
        # Periodic checkpoint
        # ---------------------

        if (
            epoch
            % checkpoint_interval
            == 0
        ):
            save_checkpoint(
                path=(
                    CHECKPOINT_DIR
                    / f"epoch_{epoch:03d}.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                config=config,
            )

        # ---------------------
        # Periodic sampling
        # ---------------------

        if (
            epoch % sample_interval
            == 0
        ):
            sample_generator = (
                torch.Generator(
                    device=device
                )
            )

            # Cùng seed ở mọi epoch:
            # sample comparison công bằng.
            sample_generator.manual_seed(
                sample_seed
            )

            samples = sample_ddpm(
                model=model,
                scheduler=scheduler,
                shape=(
                    num_sample_images,
                    3,
                    int(
                        config["data"][
                            "image_size"
                        ]
                    ),
                    int(
                        config["data"][
                            "image_size"
                        ]
                    ),
                ),
                device=device,
                generator=sample_generator,
            )

            save_sample_grid(
                samples=samples,
                path=(
                    OUTPUT_DIR
                    / (
                        f"samples_"
                        f"epoch_{epoch:03d}.png"
                    )
                ),
            )

            print(
                "  saved sample grid"
            )

        # Update curve mỗi epoch.
        save_loss_curve(
            train_losses,
            valid_losses,
        )

        history = {
            "train_loss": (
                train_losses
            ),
            "valid_loss": (
                valid_losses
            ),
            "best_valid_loss": (
                best_valid_loss
            ),
        }

        with (
            OUTPUT_DIR
            / "history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
            )

    # -------------------------
    # Last checkpoint
    # -------------------------

    save_checkpoint(
        path=(
            CHECKPOINT_DIR
            / "last.pt"
        ),
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        train_loss=train_losses[-1],
        valid_loss=valid_losses[-1],
        config=config,
    )

    print()
    print(
        "Training complete."
    )

    print(
        "Best validation loss:",
        f"{best_valid_loss:.6f}",
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_DIR / "best.pt",
    )

    print(
        "Last checkpoint:",
        CHECKPOINT_DIR / "last.pt",
    )


if __name__ == "__main__":
    main()