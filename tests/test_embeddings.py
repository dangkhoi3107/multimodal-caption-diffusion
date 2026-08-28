import torch

from src.diffusion.embeddings import (
    TimestepMLP,
    sinusoidal_timestep_embedding,
)


def test_timestep_embedding_shape():
    timesteps = torch.tensor(
        [0, 10, 100, 999],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=128,
        )
    )

    assert embedding.shape == (
        4,
        128,
    )

    assert torch.isfinite(
        embedding
    ).all()


def test_same_timestep_same_embedding():
    timesteps = torch.tensor(
        [100, 100],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=64,
        )
    )

    assert torch.allclose(
        embedding[0],
        embedding[1],
    )


def test_different_timesteps_different_embedding():
    timesteps = torch.tensor(
        [10, 900],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=64,
        )
    )

    assert not torch.allclose(
        embedding[0],
        embedding[1],
    )


def test_timestep_zero_cos_sin_structure():
    timesteps = torch.tensor(
        [0],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=8,
        )
    )

    expected = torch.tensor(
        [
            [
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )

    assert torch.allclose(
        embedding,
        expected,
    )


def test_odd_embedding_dimension():
    timesteps = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=7,
        )
    )

    assert embedding.shape == (
        3,
        7,
    )


def test_timestep_mlp_shape():
    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=128,
        )
    )

    mlp = TimestepMLP(
        embedding_dim=128,
        time_dim=256,
    )

    output = mlp(
        embedding
    )

    assert output.shape == (
        4,
        256,
    )

    assert torch.isfinite(
        output
    ).all()


def test_timestep_mlp_preserves_batch_size():
    embedding = torch.randn(
        7,
        64,
    )

    mlp = TimestepMLP(
        embedding_dim=64,
        time_dim=128,
    )

    output = mlp(
        embedding
    )

    assert output.shape[0] == 7


def test_timestep_mlp_has_gradients():
    embedding = torch.randn(
        4,
        64,
        requires_grad=True,
    )

    mlp = TimestepMLP(
        embedding_dim=64,
        time_dim=128,
    )

    output = mlp(
        embedding
    )

    loss = output.mean()

    loss.backward()

    assert embedding.grad is not None

    for parameter in mlp.parameters():
        assert parameter.grad is not None


def test_timestep_mlp_invalid_feature_dimension():
    import pytest

    mlp = TimestepMLP(
        embedding_dim=64,
        time_dim=128,
    )

    wrong_embedding = torch.randn(
        4,
        32,
    )

    with pytest.raises(
        ValueError,
        match="feature dimension",
    ):
        mlp(
            wrong_embedding
        )