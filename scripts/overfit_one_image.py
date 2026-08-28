import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.product_dataset import (
    ProductImageDataset,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)
from src.diffusion.trainer import (
    compute_diffusion_loss,
)
from src.diffusion.unet import (
    UNet,
)


CONFIG_PATH = Path(
    "configs/phase1_unconditional.yaml"
)

OUTPUT_DIR = Path(
    "outputs/phase1_overfit_one_image"
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
def evaluate_fixed_case(
    model: UNet,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    model.eval()

    loss = compute_diffusion_loss(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
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

    print(
        f"Device: {device}"
    )

    # -------------------------
    # Dataset
    # -------------------------

    dataset = ProductImageDataset(
        Path(
            config["data"][
                "train_metadata"
            ]
        )
    )

    item = dataset[0]

    x_0 = item["image"]

    # [3,64,64] -> [1,3,64,64]
    x_0 = x_0.unsqueeze(0)

    x_0 = x_0.to(device)

    print(
        "Training image:",
        item["file_name"],
    )

    print(
        "x_0 shape:",
        tuple(x_0.shape),
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

    num_timesteps = int(
        config["diffusion"][
            "num_timesteps"
        ]
    )

    steps = int(
        config["overfit_one_image"][
            "steps"
        ]
    )

    log_interval = int(
        config["overfit_one_image"][
            "log_interval"
        ]
    )

    # -------------------------
    # Fixed evaluation case
    # -------------------------
    #
    # Ta giữ nguyên t và noise này
    # trước/sau training để so sánh
    # công bằng.
    #

    fixed_timestep = torch.tensor(
        [500],
        dtype=torch.long,
        device=device,
    )

    fixed_generator = torch.Generator(
        device=device
    )

    fixed_generator.manual_seed(
        seed + 1000
    )

    fixed_noise = torch.randn(
        x_0.shape,
        device=device,
        dtype=x_0.dtype,
        generator=fixed_generator,
    )

    initial_eval_loss = (
        evaluate_fixed_case(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=fixed_timestep,
            noise=fixed_noise,
        )
    )

    print()
    print(
        "Initial fixed-case loss:",
        f"{initial_eval_loss:.6f}",
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
        # Random timestep mỗi step
        timesteps = torch.randint(
            low=0,
            high=num_timesteps,
            size=(1,),
            device=device,
            dtype=torch.long,
        )

        # Random epsilon mỗi step
        noise = torch.randn_like(
            x_0
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = compute_diffusion_loss(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Loss became NaN or Inf"
            )

        loss.backward()

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
                f"t={timesteps.item()}"
            )

    # -------------------------
    # Fixed evaluation again
    # -------------------------

    final_eval_loss = (
        evaluate_fixed_case(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            timesteps=fixed_timestep,
            noise=fixed_noise,
        )
    )

    improvement_ratio = (
        final_eval_loss
        / initial_eval_loss
    )

    print()
    print(
        "Initial fixed loss:",
        f"{initial_eval_loss:.6f}",
    )

    print(
        "Final fixed loss:",
        f"{final_eval_loss:.6f}",
    )

    print(
        "Final / initial:",
        f"{improvement_ratio:.4f}",
    )

    # -------------------------
    # Output
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(losses)

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "One-image DDPM Overfit"
    )

    plt.tight_layout()

    loss_curve_path = (
        OUTPUT_DIR
        / "loss_curve.png"
    )

    plt.savefig(
        loss_curve_path,
        dpi=150,
    )

    plt.close()

    checkpoint_path = (
        OUTPUT_DIR
        / "model.pt"
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
            "seed": seed,
            "initial_eval_loss": (
                initial_eval_loss
            ),
            "final_eval_loss": (
                final_eval_loss
            ),
            "source_file": (
                item["file_name"]
            ),
        },
        checkpoint_path,
    )

    print()
    print(
        f"Loss curve: "
        f"{loss_curve_path}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    # Correctness gate
    if final_eval_loss >= initial_eval_loss:
        raise RuntimeError(
            "One-image overfit failed: "
            "fixed evaluation loss "
            "did not improve."
        )

    print()
    print(
        "PASS: fixed noise prediction "
        "improved after training."
    )


if __name__ == "__main__":
    main()