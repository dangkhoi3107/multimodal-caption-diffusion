import matplotlib.pyplot as plt
import torch

from src.diffusion.scheduler import (
    compute_diffusion_coefficients,
    linear_beta_schedule,
    q_sample,
)
from src.diffusion.toy_data import sample_gaussian_mixture


def main():
    torch.manual_seed(42)

    x_0 = sample_gaussian_mixture(
        num_samples=5000, #5000 pixels
        seed=36,
    )

    betas = linear_beta_schedule(
        num_timesteps=1000,
    )

    _, alpha_bars = compute_diffusion_coefficients(
        betas
    )

    timesteps = [0, 100, 300, 600, 999]

    fig, axes = plt.subplots(
        1,
        len(timesteps),
        figsize=(20, 4),
    )

    for ax, t in zip(axes, timesteps):
        noise = torch.randn_like(x_0)

        x_t = q_sample(
            x_0=x_0,
            t=t,
            alpha_bars=alpha_bars,
            noise=noise,
        )

        x_t = x_t.numpy()

        ax.scatter(
            x_t[:, 0],
            x_t[:, 1],
            s=3,
            alpha=0.5,
        )

        ax.set_title(f"t = {t}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()