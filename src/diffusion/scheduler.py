"""
DDPM diffusion scheduler.

Keeps the original F0 toy helpers and adds a batch-safe DDPMScheduler
for Phase 1 image diffusion.
"""

import torch


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    if num_timesteps <= 0:
        raise ValueError(
            "num_timesteps must be positive"
        )

    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(
            "Expected 0 < beta_start < beta_end < 1"
        )

    return torch.linspace(
        beta_start,
        beta_end,
        num_timesteps,
        dtype=torch.float32,
    )


def compute_diffusion_coefficients(
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if betas.ndim != 1:
        raise ValueError(
            "betas must be a 1D tensor"
        )

    if not torch.all(
        (betas > 0.0)
        & (betas < 1.0)
    ):
        raise ValueError(
            "all beta values must be between 0 and 1"
        )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    return alphas, alpha_bars


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    """
    Select one coefficient per sample and reshape for broadcasting.

    values: [T]
    timesteps: [B]
    target_shape: [B, C, H, W]
    output: [B, 1, 1, 1]
    """
    if values.ndim != 1:
        raise ValueError(
            "values must have shape [T]"
        )

    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps must have shape [B]"
        )

    if len(target_shape) < 1:
        raise ValueError(
            "target_shape must have at least one dimension"
        )

    if timesteps.shape[0] != target_shape[0]:
        raise ValueError(
            "timesteps batch size must match target batch size"
        )

    if timesteps.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "timesteps must have integer dtype"
        )

    if torch.any(timesteps < 0):
        raise ValueError(
            "timesteps must be non-negative"
        )

    if torch.any(
        timesteps >= values.shape[0]
    ):
        raise ValueError(
            "timesteps contain out-of-range values"
        )

    values = values.to(
        device=timesteps.device
    )

    selected = values[
        timesteps.long()
    ]

    broadcast_shape = (
        timesteps.shape[0],
        *(
            [1]
            * (len(target_shape) - 1)
        ),
    )

    return selected.reshape(
        broadcast_shape
    )


# ---------------------------------------------------------------------
# F0 helpers: scalar timestep for the whole tensor.
# ---------------------------------------------------------------------

def q_sample(
    x_0: torch.Tensor,
    t: int,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if t < 0 or t >= alpha_bars.shape[0]:
        raise ValueError(
            "t is out of range"
        )

    if noise is None:
        noise = torch.randn_like(x_0)

    if noise.shape != x_0.shape:
        raise ValueError(
            "noise must have the same shape as x_0"
        )

    alpha_bar_t = alpha_bars[t].to(
        device=x_0.device,
        dtype=x_0.dtype,
    )

    return (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(
            1.0 - alpha_bar_t
        ) * noise
    )


def q_sample_iterative(
    x_0: torch.Tensor,
    t: int,
    alphas: torch.Tensor,
) -> torch.Tensor:
    if t < 0 or t >= alphas.shape[0]:
        raise ValueError(
            "t is out of range"
        )

    x = x_0.clone()

    for step in range(t + 1):
        noise = torch.randn_like(x)

        alpha_t = alphas[step].to(
            device=x.device,
            dtype=x.dtype,
        )

        x = (
            torch.sqrt(alpha_t) * x
            + torch.sqrt(
                1.0 - alpha_t
            ) * noise
        )

    return x


class DDPMScheduler:
    """
    DDPM scheduler for batched image diffusion.

    Phase 1 contracts:
        x_0: [B, C, H, W]
        timesteps: [B]
        noise: [B, C, H, W]
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.num_timesteps = num_timesteps

        self.betas = linear_beta_schedule(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        (
            self.alphas,
            self.alpha_bars,
        ) = compute_diffusion_coefficients(
            self.betas
        )

        self.sqrt_alpha_bars = torch.sqrt(
            self.alpha_bars
        )

        self.sqrt_one_minus_alpha_bars = (
            torch.sqrt(
                1.0 - self.alpha_bars
            )
        )

        # alpha_bar_{t-1}; convention alpha_bar_{-1} = 1.
        self.alpha_bars_prev = torch.cat(
            [
                torch.ones(
                    1,
                    dtype=self.alpha_bars.dtype,
                ),
                self.alpha_bars[:-1],
            ],
            dim=0,
        )

        # Posterior variance:
        #
        # beta_tilde_t =
        # beta_t * (1 - alpha_bar_{t-1})
        #        / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas
            * (
                1.0
                - self.alpha_bars_prev
            )
            / (
                1.0
                - self.alpha_bars
            )
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_0.ndim != 4:
            raise ValueError(
                "x_0 must have shape [B, C, H, W]"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [B]"
            )

        if timesteps.shape[0] != x_0.shape[0]:
            raise ValueError(
                "timesteps batch size must match x_0 batch size"
            )

        if timesteps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "timesteps must have integer dtype"
            )

        if noise is None:
            noise = torch.randn_like(x_0)

        if noise.shape != x_0.shape:
            raise ValueError(
                "noise must have the same shape as x_0"
            )

        timesteps = timesteps.to(
            device=x_0.device
        )

        sqrt_alpha_bar_t = extract(
            values=self.sqrt_alpha_bars.to(
                device=x_0.device,
                dtype=x_0.dtype,
            ),
            timesteps=timesteps,
            target_shape=x_0.shape,
        )

        sqrt_one_minus_alpha_bar_t = extract(
            values=self.sqrt_one_minus_alpha_bars.to(
                device=x_0.device,
                dtype=x_0.dtype,
            ),
            timesteps=timesteps,
            target_shape=x_0.shape,
        )

        return (
            sqrt_alpha_bar_t * x_0
            + sqrt_one_minus_alpha_bar_t * noise
        )

    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if x_t.ndim != 4:
            raise ValueError(
                "x_t must have shape [B, C, H, W]"
            )

        if model_output.shape != x_t.shape:
            raise ValueError(
                "model_output must have the same shape as x_t"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [B]"
            )

        if timesteps.shape[0] != x_t.shape[0]:
            raise ValueError(
                "timesteps batch size must match x_t"
            )

        if timesteps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "timesteps must have integer dtype"
            )

        timesteps = timesteps.to(
            device=x_t.device
        )

        betas_t = extract(
            values=self.betas.to(
                device=x_t.device,
                dtype=x_t.dtype,
            ),
            timesteps=timesteps,
            target_shape=x_t.shape,
        )

        alphas_t = extract(
            values=self.alphas.to(
                device=x_t.device,
                dtype=x_t.dtype,
            ),
            timesteps=timesteps,
            target_shape=x_t.shape,
        )

        alpha_bars_t = extract(
            values=self.alpha_bars.to(
                device=x_t.device,
                dtype=x_t.dtype,
            ),
            timesteps=timesteps,
            target_shape=x_t.shape,
        )

        posterior_variance_t = extract(
            values=self.posterior_variance.to(
                device=x_t.device,
                dtype=x_t.dtype,
            ),
            timesteps=timesteps,
            target_shape=x_t.shape,
        )

        # DDPM epsilon-prediction reverse mean:
        #
        # mu_theta =
        # 1 / sqrt(alpha_t)
        # * (
        #     x_t
        #     - beta_t / sqrt(1 - alpha_bar_t)
        #       * epsilon_theta(x_t, t)
        # )
        model_mean = (
            1.0
            / torch.sqrt(alphas_t)
        ) * (
            x_t
            - (
                betas_t
                / torch.sqrt(
                    1.0 - alpha_bars_t
                )
            )
            * model_output
        )

        return (
            model_mean,
            posterior_variance_t,
        )

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [B]"
            )

        if timesteps.shape[0] != x_t.shape[0]:
            raise ValueError(
                "timesteps batch size must match x_t"
            )

        if timesteps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "timesteps must have integer dtype"
            )

        timesteps = timesteps.to(
            device=x_t.device
        )

        model_output = model(
            x_t,
            timesteps,
        )

        (
            model_mean,
            posterior_variance,
        ) = self.p_mean_variance(
            model_output=model_output,
            x_t=x_t,
            timesteps=timesteps,
        )

        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )

        # No random noise when t == 0.
        nonzero_mask = (
            timesteps > 0
        ).to(
            dtype=x_t.dtype
        ).reshape(
            x_t.shape[0],
            *(
                [1]
                * (x_t.ndim - 1)
            ),
        )

        return (
            model_mean
            + nonzero_mask
            * torch.sqrt(
                posterior_variance
            )
            * noise
        )
