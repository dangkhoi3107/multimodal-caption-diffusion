import math

import torch

from src.diffusion.toy_ode import euler_integrate


def test_euler_trajectory_shape():
    x_0 = torch.tensor([5.0])

    trajectory = euler_integrate(
        x_0=x_0,
        dt=0.1,
        num_steps=10,
    )

    assert trajectory.shape == (11, 1)

    assert torch.allclose(
        trajectory[0],
        x_0,
    )


def test_euler_values_are_finite():
    x_0 = torch.tensor([5.0])

    trajectory = euler_integrate(
        x_0=x_0,
        dt=0.1,
        num_steps=10,
    )

    assert torch.isfinite(trajectory).all()


def test_euler_error_decreases_with_smaller_dt():
    x_0 = torch.tensor([5.0])

    final_time = 1.0

    exact = x_0 * math.exp(-final_time)

    dt_large = 0.2
    steps_large = int(final_time / dt_large)

    dt_small = 0.01
    steps_small = int(final_time / dt_small)

    trajectory_large = euler_integrate(
        x_0=x_0,
        dt=dt_large,
        num_steps=steps_large,
    )

    trajectory_small = euler_integrate(
        x_0=x_0,
        dt=dt_small,
        num_steps=steps_small,
    )

    error_large = torch.abs(
        trajectory_large[-1] - exact
    )

    error_small = torch.abs(
        trajectory_small[-1] - exact
    )

    assert torch.all(
        error_small < error_large
    )