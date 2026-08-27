# F0 Math and Toy Distributions

## Hypothesis

- Gaussian mixture statistics should be finite and reproducible with a fixed seed.
- Forward diffusion should gradually destroy the two-cluster structure.
- At large timesteps, the distribution should approach standard Gaussian noise.
- The analytic Gaussian score should point toward regions of higher probability density.
- Euler integration error should decrease when the step size becomes smaller.

## Configuration

- Seed: 42
- Number of Gaussian mixture samples: 5000
- Diffusion timesteps: 1000
- Beta start: 0.0001
- Beta end: 0.02
- Visualization timesteps: [0, 100, 300, 600, 999]

## Gaussian Mixture Statistics

Empirical mean:

```text
[-0.016188830137252808, 0.009202873334288597]
```

Empirical covariance:

```text
[[4.509042739868164, -0.010484142228960991], [-0.010484142228960991, 0.4950045049190521]]
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
{
  "0.2": 0.07301349192857742,
  "0.1": 0.03697947412729263,
  "0.01": 0.003731049597263336
}
```

Smaller step sizes should produce smaller numerical error.

## Formula to Tensor Mapping

| Math symbol | Tensor name | Shape | Meaning |
|---|---|---|---|
| $x_0$ | `x_0` | `[N, 2]` | Clean Gaussian-mixture samples |
| $eta_t$ | `betas[t]` | scalar | Noise variance added at timestep `t` |
| $lpha_t$ | `alphas[t]` | scalar | Signal retention at timestep `t` |
| $ar{lpha}_t$ | `alpha_bars[t]` | scalar | Cumulative signal retention from step 0 to `t` |
| $\epsilon$ | `noise` | `[N, 2]` | Standard Gaussian noise |
| $x_t$ | `x_t` | `[N, 2]` | Noised samples at timestep `t` |
| $\mu$ | `mean` | `[2]` | Gaussian mean |
| $\Sigma$ | `covariance` | `[2, 2]` | Gaussian covariance matrix |
| $
abla_x \log p(x)$ | `scores` | `[N, 2]` | Score vector |

## Gate

Final PASS/FAIL is determined after running the complete F0 test suite.

## Limitations

- F0 uses synthetic 2D distributions rather than images.
- No neural network is trained in this checkpoint.
- The score visualization currently uses a standard Gaussian distribution.
- Forward diffusion is tested as a known mathematical process.
- Reverse diffusion has not been implemented yet.
