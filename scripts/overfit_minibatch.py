import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.product_dataset import ProductImageDataset
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.trainer import compute_diffusion_loss
from src.diffusion.unet import UNet


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

OUTPUT_DIR = Path(
    "outputs/phase1_overfit_minibatch"
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


@torch.no_grad()
def fixed_evaluation(
    model,
    scheduler,
    images,
    timesteps,
    noise,
) -> float:
    model.eval()

    loss = compute_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=images,
        timesteps=timesteps,
        noise=noise,
    )

    return float(loss.item())


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

    print("Device:", device)

    # -------------------------
    # Dataset
    # -------------------------

    dataset = ProductImageDataset(
        Path(
            config["data"]["train_metadata"]
        )
    )

    num_images = int(
        config["overfit_minibatch"][
            "num_images"
        ]
    )

    if num_images > len(dataset):
        raise ValueError(
            "num_images exceeds dataset size"
        )

    images = torch.stack(
        [
            dataset[index]["image"]
            for index in range(num_images)
        ],
        dim=0,
    ).to(device)

    print(
        "Mini-batch shape:",
        tuple(images.shape),
    )

    print("Files:")

    for index in range(num_images):
        print(
            "-",
            dataset[index]["file_name"],
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
            config["model"]["in_channels"]
        ),
        out_channels=int(
            config["model"]["out_channels"]
        ),
        base_channels=int(
            config["model"]["base_channels"]
        ),
        time_embedding_dim=int(
            config["model"][
                "time_embedding_dim"
            ]
        ),
        time_dim=int(
            config["model"]["time_dim"]
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"][
                "learning_rate"
            ]
        ),
    )

    num_timesteps = int(
        config["diffusion"][
            "num_timesteps"
        ]
    )

    steps = int(
        config["overfit_minibatch"][
            "steps"
        ]
    )

    log_interval = int(
        config["overfit_minibatch"][
            "log_interval"
        ]
    )

    # -------------------------
    # Fixed evaluation case
    # -------------------------

    fixed_generator = torch.Generator(
        device=device
    )

    fixed_generator.manual_seed(
        seed + 2000
    )

    fixed_timesteps = torch.randint(
        low=0,
        high=num_timesteps,
        size=(num_images,),
        device=device,
        dtype=torch.long,
        generator=fixed_generator,
    )

    fixed_noise = torch.randn(
        images.shape,
        device=device,
        dtype=images.dtype,
        generator=fixed_generator,
    )

    initial_fixed_loss = fixed_evaluation(
        model=model,
        scheduler=scheduler,
        images=images,
        timesteps=fixed_timesteps,
        noise=fixed_noise,
    )

    print()
    print(
        "Initial fixed loss:",
        f"{initial_fixed_loss:.6f}",
    )

    # -------------------------
    # Training
    # -------------------------

    losses = []

    model.train()

    for step in range(
        1,
        steps + 1,
    ):
        timesteps = torch.randint(
            low=0,
            high=num_timesteps,
            size=(num_images,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(
            images
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = compute_diffusion_loss(
            model=model,
            scheduler=scheduler,
            x_0=images,
            timesteps=timesteps,
            noise=noise,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Loss became NaN or Inf"
            )

        loss.backward()

        for name, parameter in (
            model.named_parameters()
        ):
            if parameter.grad is None:
                continue

            if not torch.isfinite(
                parameter.grad
            ).all():
                raise RuntimeError(
                    f"Non-finite gradient: {name}"
                )

        optimizer.step()

        loss_value = float(
            loss.detach().item()
        )

        losses.append(
            loss_value
        )

        if (
            step == 1
            or step % log_interval == 0
        ):
            print(
                f"step={step:04d} "
                f"loss={loss_value:.6f} "
                f"t_min={timesteps.min().item()} "
                f"t_max={timesteps.max().item()}"
            )

    # -------------------------
    # Final fixed evaluation
    # -------------------------

    final_fixed_loss = fixed_evaluation(
        model=model,
        scheduler=scheduler,
        images=images,
        timesteps=fixed_timesteps,
        noise=fixed_noise,
    )

    ratio = (
        final_fixed_loss
        / initial_fixed_loss
    )

    print()
    print(
        "Initial fixed loss:",
        f"{initial_fixed_loss:.6f}",
    )

    print(
        "Final fixed loss:",
        f"{final_fixed_loss:.6f}",
    )

    print(
        "Final / initial:",
        f"{ratio:.4f}",
    )

    # -------------------------
    # Save
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        losses,
        alpha=0.5,
        label="step loss",
    )

    # Moving average để nhìn trend rõ hơn.
    window = 50

    if len(losses) >= window:
        kernel = np.ones(window) / window

        moving_average = np.convolve(
            losses,
            kernel,
            mode="valid",
        )

        plt.plot(
            range(
                window - 1,
                len(losses),
            ),
            moving_average,
            label="50-step moving average",
        )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "8-image DDPM Overfit"
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR / "loss_curve.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    checkpoint_path = (
        OUTPUT_DIR / "model.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "steps": steps,
            "num_images": num_images,
            "seed": seed,
            "initial_fixed_loss": (
                initial_fixed_loss
            ),
            "final_fixed_loss": (
                final_fixed_loss
            ),
        },
        checkpoint_path,
    )

    print()
    print("Plot:", plot_path)
    print("Checkpoint:", checkpoint_path)

    if (
        final_fixed_loss
        >= initial_fixed_loss
    ):
        raise RuntimeError(
            "Mini-batch overfit failed"
        )

    print()
    print(
        "PASS: mini-batch fixed "
        "evaluation improved."
    )


if __name__ == "__main__":
    main()