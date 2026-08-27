"""
Scheduler thêm noise
"""
import torch


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4, #0.0001
    beta_end: float = 2e-2, #0.02
) -> torch.Tensor:
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")

    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(
            "Expected 0 < beta_start < beta_end < 1"
        )

    betas = torch.linspace(
        beta_start,
        beta_end,
        num_timesteps,
        dtype=torch.float32,
    )

    return betas

""" Tính các hệ số khuếch tán"""
def compute_diffusion_coefficients(
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if betas.ndim != 1:
        raise ValueError("betas must be a 1D tensor")

    if not torch.all((betas > 0.0) & (betas < 1.0)):
        raise ValueError("all beta values must be between 0 and 1")

    """
    betas: nhiễu tại thời điểm t
    alphas: tín hiệu còn lại tại thời điểm t
    """
    alphas = 1.0 - betas 


    """
    Cumulative Product - Tích lũy kế
    
    alpha_bars = alpha1 x alpha2 x ... 

    Càng thêm nhiễu thì alpha càng tiệm cận về 0
    """
    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    return alphas, alpha_bars


def q_sample(
    x_0: torch.Tensor,
    t: int,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if t < 0 or t >= alpha_bars.shape[0]:
        raise ValueError("t is out of range")

    if noise is None:
        noise = torch.randn_like(x_0) # Standard noise: Tự tạo noise có shape = ảnh gốc 

    if noise.shape != x_0.shape:
        raise ValueError("noise must have the same shape as x_0")

    """
    lấy giá trị tại thời điểm t
    """
    alpha_bar_t = alpha_bars[t]

    """
    diffusion forward formula
    Cần căn bậc 2 vì khi cộng 2 biến độc lập thì phương sai (Variance) sẽ tăng theo hệ số nhân
    """
    x_t = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    return x_t


def q_sample_iterative(
    x_0: torch.Tensor,
    t: int,
    alphas: torch.Tensor,
) -> torch.Tensor:
    if t < 0 or t >= alphas.shape[0]:
        raise ValueError("t is out of range")

    x = x_0.clone()

    for step in range(t + 1):
        noise = torch.randn_like(x)

        alpha_t = alphas[step]

        x = (
            torch.sqrt(alpha_t) * x
            + torch.sqrt(1.0 - alpha_t) * noise
        )

    return x