import torch


def euler_integrate(
    x_0: torch.Tensor,
    dt: float,
    num_steps: int,
) -> torch.Tensor:
    if dt <= 0:
        raise ValueError("dt must be positive")

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    x = x_0.clone()

    trajectory = [x.clone()]

    for _ in range(num_steps):
        dx_dt = -x # độ biến thiên luôn ngược chiều giá trị hiện tại, truy vấn lùi, hướng cho forward

        x = x + dt * dx_dt # x = x - x*dt

        trajectory.append(x.clone())

    return torch.stack(trajectory)