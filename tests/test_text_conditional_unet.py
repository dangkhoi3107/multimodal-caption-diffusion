import torch

from src.diffusion.text_conditional_unet import (
    TextConditionalUNet,
)
from src.diffusion.text_conditioning import (
    TextConditioner,
)


def make_model():
    return TextConditionalUNet(
        vocab_size=19,
        pad_id=0,
        max_length=10,
        text_embedding_dim=128,
        text_num_heads=4,
        text_num_layers=2,
        text_feedforward_dim=256,
        text_dropout=0.0,
        base_channels=32,
        time_embedding_dim=64,
        time_dim=128,
    )


def make_text_batch():
    token_ids = torch.tensor(
        [
            [
                1, 4, 15, 12, 10,
                14, 2, 0, 0, 0,
            ],
            [
                1, 4, 5, 9, 8,
                17, 2, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    mask = (
        token_ids != 0
    )

    return (
        token_ids,
        mask,
    )


def test_text_conditioner_shape():
    conditioner = TextConditioner(
        text_dim=128,
        condition_dim=256,
    )

    x = torch.randn(
        3,
        128,
    )

    output = conditioner(
        x
    )

    assert output.shape == (
        3,
        256,
    )


def test_output_shape():
    model = make_model()

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [100, 700],
        dtype=torch.long,
    )

    token_ids, mask = (
        make_text_batch()
    )

    output = model(
        x_t,
        timesteps,
        token_ids,
        mask,
    )

    assert output.shape == (
        2,
        3,
        64,
        64,
    )

    assert torch.isfinite(
        output
    ).all()


def test_different_text_changes_output():
    torch.manual_seed(
        42
    )

    model = make_model()
    model.eval()

    # Identical image/noise and timestep.
    x = torch.randn(
        1,
        3,
        64,
        64,
    )

    x_t = x.repeat(
        2,
        1,
        1,
        1,
    )

    timesteps = torch.tensor(
        [500, 500],
        dtype=torch.long,
    )

    token_ids, mask = (
        make_text_batch()
    )

    with torch.no_grad():
        output = model(
            x_t,
            timesteps,
            token_ids,
            mask,
        )

    difference = (
        output[0]
        - output[1]
    ).abs().mean()

    assert (
        difference.item()
        > 0.0
    )


def test_empty_prompt_forward():
    model = make_model()

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [100, 200],
        dtype=torch.long,
    )

    # Empty prompt encoded as:
    # BOS EOS PAD PAD ...
    token_ids = torch.tensor(
        [
            [
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            [
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
        ],
        dtype=torch.long,
    )

    mask = (
        token_ids != 0
    )

    output = model(
        x_t,
        timesteps,
        token_ids,
        mask,
    )

    assert output.shape == (
        2,
        3,
        64,
        64,
    )

    assert torch.isfinite(
        output
    ).all()


def test_gradient_reaches_text_encoder():
    model = make_model()

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [50, 900],
        dtype=torch.long,
    )

    token_ids, mask = (
        make_text_batch()
    )

    output = model(
        x_t,
        timesteps,
        token_ids,
        mask,
    )

    loss = (
        output ** 2
    ).mean()

    loss.backward()

    gradient = (
        model
        .text_encoder
        .token_embedding
        .weight
        .grad
    )

    assert gradient is not None

    assert torch.isfinite(
        gradient
    ).all()

    assert (
        gradient.abs().sum().item()
        > 0.0
    )


def test_gradient_reaches_text_conditioner():
    model = make_model()

    x_t = torch.randn(
        2,
        3,
        64,
        64,
    )

    timesteps = torch.tensor(
        [100, 400],
        dtype=torch.long,
    )

    token_ids, mask = (
        make_text_batch()
    )

    output = model(
        x_t,
        timesteps,
        token_ids,
        mask,
    )

    output.mean().backward()

    gradients = [
        parameter.grad
        for parameter
        in model.text_conditioner.parameters()
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert all(
        torch.isfinite(
            gradient
        ).all()
        for gradient in gradients
    )