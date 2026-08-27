import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.diffusion.scheduler import (
    compute_diffusion_coefficients,
    linear_beta_schedule,
    q_sample,
)
from src.diffusion.toy_data import sample_gaussian_mixture
from src.diffusion.toy_ode import euler_integrate
from src.diffusion.toy_score import gaussian_score


def empirical_covariance(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0)
    centered = x - mean
    return centered.T @ centered / (x.shape[0] - 1)


def main():
    config_path = Path("configs/experiments/f0_math_toy.yaml")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    torch.manual_seed(seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / "f0_math_toy" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    resolved_config_path = output_dir / "config.resolved.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    num_samples = config["gaussian_mixture"]["num_samples"]
    x_0 = sample_gaussian_mixture(
        num_samples=num_samples,
        seed=seed,
    )

    empirical_mean = x_0.mean(dim=0)
    empirical_cov = empirical_covariance(x_0)

    statistics = {
        "num_samples": num_samples,
        "empirical_mean": empirical_mean.tolist(),
        "empirical_covariance": empirical_cov.tolist(),
    }

    with (output_dir / "gaussian_statistics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(statistics, f, indent=2)

    diffusion_config = config["diffusion"]

    betas = linear_beta_schedule(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )

    _, alpha_bars = compute_diffusion_coefficients(betas)

    timesteps = diffusion_config["plot_timesteps"]

    fig, axes = plt.subplots(
        1,
        len(timesteps),
        figsize=(20, 4),
    )

    for ax, t in zip(axes, timesteps):
        if t == 0:
            x_t = x_0
        else:
            noise = torch.randn_like(x_0)
            x_t = q_sample(
                x_0=x_0,
                t=t,
                alpha_bars=alpha_bars,
                noise=noise,
            )

        points = x_t.detach().cpu().numpy()

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=3,
            alpha=0.5,
        )

        ax.set_title(f"t = {t}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "forward_noising_2d.png", dpi=150)
    plt.close(fig)

    score_config = config["score"]

    grid_values = torch.linspace(
        score_config["grid_min"],
        score_config["grid_max"],
        score_config["grid_points"],
    )

    grid_x, grid_y = torch.meshgrid(
        grid_values,
        grid_values,
        indexing="xy",
    )

    grid_points = torch.stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
        ],
        dim=1,
    )

    mean = torch.zeros(2, dtype=torch.float32)
    covariance = torch.eye(2, dtype=torch.float32)

    scores = gaussian_score(
        x=grid_points,
        mean=mean,
        covariance=covariance,
    )

    fig = plt.figure(figsize=(6, 6))

    plt.quiver(
        grid_points[:, 0].numpy(),
        grid_points[:, 1].numpy(),
        scores[:, 0].numpy(),
        scores[:, 1].numpy(),
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Gaussian Score Field")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "score_field_2d.png", dpi=150)
    plt.close(fig)

    ode_config = config["ode"]

    x_initial = torch.tensor(
        [ode_config["x0"]],
        dtype=torch.float32,
    )

    final_time = ode_config["final_time"]

    fig = plt.figure(figsize=(7, 5))

    exact_times = torch.linspace(
        0.0,
        final_time,
        300,
    )

    exact_values = x_initial.item() * torch.exp(-exact_times)

    plt.plot(
        exact_times.numpy(),
        exact_values.numpy(),
        label="Exact",
    )

    ode_errors = {}

    for dt in ode_config["step_sizes"]:
        num_steps = int(final_time / dt)

        trajectory = euler_integrate(
            x_0=x_initial,
            dt=dt,
            num_steps=num_steps,
        )

        times = torch.arange(
            num_steps + 1,
            dtype=torch.float32,
        ) * dt

        plt.plot(
            times.numpy(),
            trajectory[:, 0].numpy(),
            label=f"Euler dt={dt}",
        )

        actual_final_time = num_steps * dt
        exact_final_value = (
            x_initial.item()
            * torch.exp(torch.tensor(-actual_final_time)).item()
        )
        numerical_final_value = trajectory[-1, 0].item()
        error = abs(numerical_final_value - exact_final_value)
        ode_errors[str(dt)] = error

    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Euler ODE Integration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ode_trajectory.png", dpi=150)
    plt.close(fig)

    summary = rf"""# F0 Math and Toy Distributions

## Hypothesis

- Gaussian mixture statistics should be finite and reproducible with a fixed seed.
- Forward diffusion should gradually destroy the two-cluster structure.
- At large timesteps, the distribution should approach standard Gaussian noise.
- The analytic Gaussian score should point toward regions of higher probability density.
- Euler integration error should decrease when the step size becomes smaller.

## Configuration

- Seed: {seed}
- Number of Gaussian mixture samples: {num_samples}
- Diffusion timesteps: {diffusion_config["num_timesteps"]}
- Beta start: {diffusion_config["beta_start"]}
- Beta end: {diffusion_config["beta_end"]}
- Visualization timesteps: {timesteps}

## Gaussian Mixture Statistics

Empirical mean:

```text
{empirical_mean.tolist()}
```

Empirical covariance:

```text
{empirical_cov.tolist()}
```

## Forward Diffusion

The forward-noising visualization is saved as:

```text
forward_noising_2d.png
```

The original two-cluster structure gradually disappears as the timestep increases.

The forward process uses:

```text
x_t = sqrt(alpha_bar_t) * x_0
      + sqrt(1 - alpha_bar_t) * noise
```

At large timesteps, the contribution from `x_0` becomes small and Gaussian noise dominates.

## Score Function

The Gaussian score vector field is saved as:

```text
score_field_2d.png
```

For a standard Gaussian:

```text
score(x) = -x
```

Therefore the score vectors point toward the origin, where probability density is higher.

## ODE

Euler trajectories are compared with the exact solution of:

```text
dx/dt = -x
```

The exact solution is:

```text
x(t) = x_0 * exp(-t)
```

The result is saved as:

```text
ode_trajectory.png
```

Euler final-value errors:

```text
{json.dumps(ode_errors, indent=2)}
```

Smaller step sizes should produce smaller numerical error.

## Formula to Tensor Mapping

| Math symbol | Tensor name | Shape | Meaning |
|---|---|---|---|
| $x_0$ | `x_0` | `[N, 2]` | Clean Gaussian-mixture samples |
| $\beta_t$ | `betas[t]` | scalar | Noise variance added at timestep `t` |
| $\alpha_t$ | `alphas[t]` | scalar | Signal retention at timestep `t` |
| $\bar{{\alpha}}_t$ | `alpha_bars[t]` | scalar | Cumulative signal retention from step 0 to `t` |
| $\epsilon$ | `noise` | `[N, 2]` | Standard Gaussian noise |
| $x_t$ | `x_t` | `[N, 2]` | Noised samples at timestep `t` |
| $\mu$ | `mean` | `[2]` | Gaussian mean |
| $\Sigma$ | `covariance` | `[2, 2]` | Gaussian covariance matrix |
| $\nabla_x \log p(x)$ | `scores` | `[N, 2]` | Score vector |

## Gate

Final PASS/FAIL is determined after running the complete F0 test suite.

## Limitations

- F0 uses synthetic 2D distributions rather than images.
- No neural network is trained in this checkpoint.
- The score visualization currently uses a standard Gaussian distribution.
- Forward diffusion is tested as a known mathematical process.
- Reverse diffusion has not been implemented yet.
"""

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(summary)

    print(f"F0 artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
