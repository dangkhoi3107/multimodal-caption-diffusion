import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.data.product_dataset import (
    ProductImageDataset,
)
from src.diffusion.conditional_trainer import (
    conditional_training_step,
)
from src.diffusion.conditional_unet import (
    ConditionalUNet,
)
from src.diffusion.scheduler import (
    DDPMScheduler,
)


CONFIG_PATH = Path(
    "configs/phase2_class_conditional.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/phase2_overfit_minibatch"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def select_balanced_indices(
    dataset: ProductImageDataset,
    num_classes: int,
    images_per_class: int,
    seed: int,
) -> list[int]:
    indices_by_class = {
        class_id: []
        for class_id in range(
            num_classes
        )
    }

    for index, record in enumerate(
        dataset.records
    ):
        class_id = int(
            record["class_id"]
        )

        if class_id in indices_by_class:
            indices_by_class[
                class_id
            ].append(index)

    rng = random.Random(
        seed
    )

    selected = []

    for class_id in range(
        num_classes
    ):
        candidates = (
            indices_by_class[
                class_id
            ]
        )

        if len(candidates) < (
            images_per_class
        ):
            raise RuntimeError(
                f"class {class_id} has only "
                f"{len(candidates)} samples"
            )

        rng.shuffle(
            candidates
        )

        selected.extend(
            candidates[
                :images_per_class
            ]
        )

    return selected


def build_fixed_batch(
    dataset: ProductImageDataset,
    indices: list[int],
    device: torch.device,
):
    samples = [
        dataset[index]
        for index in indices
    ]

    images = torch.stack(
        [
            sample["image"]
            for sample in samples
        ],
        dim=0,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    class_ids = torch.stack(
        [
            sample["class_id"]
            for sample in samples
        ],
        dim=0,
    ).to(
        device=device,
        dtype=torch.long,
    )

    class_names = [
        sample["class_name"]
        for sample in samples
    ]

    file_names = [
        sample["file_name"]
        for sample in samples
    ]

    return (
        images,
        class_ids,
        class_names,
        file_names,
    )


def main():
    config = load_config()

    seed = int(
        config["training"]["seed"]
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ---------------------------------
    # Dataset
    # ---------------------------------

    dataset = ProductImageDataset(
        Path(
            config[
                "data"
            ][
                "train_metadata"
            ]
        )
    )

    num_classes = int(
        config["data"]["num_classes"]
    )

    images_per_class = int(
        config[
            "overfit_minibatch"
        ][
            "images_per_class"
        ]
    )

    selected_indices = (
        select_balanced_indices(
            dataset=dataset,
            num_classes=num_classes,
            images_per_class=(
                images_per_class
            ),
            seed=seed,
        )
    )

    (
        x_0,
        class_ids,
        class_names,
        file_names,
    ) = build_fixed_batch(
        dataset=dataset,
        indices=selected_indices,
        device=device,
    )

    print(
        "Fixed batch shape:",
        tuple(
            x_0.shape
        ),
    )

    print(
        "Class IDs:",
        class_ids.tolist(),
    )

    print(
        "Class names:"
    )

    for class_id, name in zip(
        class_ids.tolist(),
        class_names,
    ):
        print(
            f"  {class_id}: {name}"
        )

    expected_class_ids = []

    for class_id in range(
        num_classes
    ):
        expected_class_ids.extend(
            [
                class_id
            ]
            * images_per_class
        )

    if class_ids.tolist() != (
        expected_class_ids
    ):
        raise RuntimeError(
            "Balanced mini-batch "
            "construction failed"
        )

    # ---------------------------------
    # Scheduler
    # ---------------------------------

    scheduler = DDPMScheduler(
        num_timesteps=int(
            config[
                "diffusion"
            ][
                "num_timesteps"
            ]
        ),
        beta_start=float(
            config[
                "diffusion"
            ][
                "beta_start"
            ]
        ),
        beta_end=float(
            config[
                "diffusion"
            ][
                "beta_end"
            ]
        ),
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = ConditionalUNet(
        num_classes=num_classes,
        in_channels=int(
            config[
                "model"
            ][
                "in_channels"
            ]
        ),
        out_channels=int(
            config[
                "model"
            ][
                "out_channels"
            ]
        ),
        base_channels=int(
            config[
                "model"
            ][
                "base_channels"
            ]
        ),
        time_embedding_dim=int(
            config[
                "model"
            ][
                "time_embedding_dim"
            ]
        ),
        time_dim=int(
            config[
                "model"
            ][
                "time_dim"
            ]
        ),
    ).to(
        device
    )

    print(
        "Null class ID:",
        model.null_class_id,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config[
                "training"
            ][
                "learning_rate"
            ]
        ),
    )

    # IMPORTANT:
    # Generator must live on the same
    # device used for random tensors.
    generator = torch.Generator(
        device=device.type
    )

    generator.manual_seed(
        seed + 1000
    )

    condition_dropout = float(
        config[
            "training"
        ][
            "condition_dropout"
        ]
    )

    steps = int(
        config[
            "overfit_minibatch"
        ][
            "steps"
        ]
    )

    log_interval = int(
        config[
            "overfit_minibatch"
        ][
            "log_interval"
        ]
    )

    # ---------------------------------
    # Training
    # ---------------------------------

    model.train()

    losses = []
    dropped_fractions = []

    print()
    print(
        "Starting Phase 2 "
        "balanced mini-batch overfit..."
    )

    for step in range(
        1,
        steps + 1,
    ):
        metrics = (
            conditional_training_step(
                model=model,
                scheduler=scheduler,
                x_0=x_0,
                class_ids=class_ids,
                optimizer=optimizer,
                condition_dropout=(
                    condition_dropout
                ),
                null_class_id=(
                    model.null_class_id
                ),
                generator=generator,
            )
        )

        losses.append(
            metrics["loss"]
        )

        dropped_fractions.append(
            metrics[
                "dropped_fraction"
            ]
        )

        if (
            step == 1
            or step % log_interval == 0
        ):
            recent = losses[
                -min(
                    log_interval,
                    len(losses),
                ):
            ]

            mean_recent = sum(
                recent
            ) / len(
                recent
            )

            print(
                f"step={step:04d} "
                f"loss={metrics['loss']:.6f} "
                f"mean_recent={mean_recent:.6f} "
                f"dropped="
                f"{metrics['dropped_fraction']:.3f}"
            )

    # ---------------------------------
    # Evidence
    # ---------------------------------

    window = min(
        100,
        len(losses) // 4,
    )

    initial_mean = sum(
        losses[:window]
    ) / window

    final_mean = sum(
        losses[-window:]
    ) / window

    loss_ratio = (
        final_mean
        / initial_mean
    )

    overall_dropout = sum(
        dropped_fractions
    ) / len(
        dropped_fractions
    )

    print()
    print("=" * 80)

    print(
        "Initial mean loss:",
        initial_mean,
    )

    print(
        "Final mean loss:",
        final_mean,
    )

    print(
        "Final / initial:",
        loss_ratio,
    )

    print(
        "Mean dropout fraction:",
        overall_dropout,
    )

    if not torch.isfinite(
        torch.tensor(
            final_mean
        )
    ):
        raise FloatingPointError(
            "Final loss is not finite"
        )

    # ---------------------------------
    # Save artifacts
    # ---------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        OUTPUT_ROOT
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
            "num_classes": (
                num_classes
            ),
            "null_class_id": (
                model.null_class_id
            ),
            "condition_dropout": (
                condition_dropout
            ),
            "steps": steps,
            "seed": seed,
            "config": config,
        },
        checkpoint_path,
    )

    report = {
        "device": str(
            device
        ),
        "batch_size": int(
            x_0.shape[0]
        ),
        "class_ids": (
            class_ids
            .detach()
            .cpu()
            .tolist()
        ),
        "class_names": (
            class_names
        ),
        "file_names": (
            file_names
        ),
        "steps": steps,
        "condition_dropout": (
            condition_dropout
        ),
        "initial_mean_loss": (
            initial_mean
        ),
        "final_mean_loss": (
            final_mean
        ),
        "loss_ratio": (
            loss_ratio
        ),
        "mean_dropout_fraction": (
            overall_dropout
        ),
    }

    with (
        OUTPUT_ROOT
        / "report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        losses
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "MSE loss"
    )

    plt.title(
        "Phase 2 balanced "
        "mini-batch overfit"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "loss_curve.png",
        dpi=150,
    )

    plt.close()

    print()
    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Report:",
        OUTPUT_ROOT
        / "report.json",
    )

    print(
        "Loss curve:",
        OUTPUT_ROOT
        / "loss_curve.png",
    )


if __name__ == "__main__":
    main()