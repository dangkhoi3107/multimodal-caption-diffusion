import pytest
import torch

from src.diffusion.conditioning import (
    ClassConditioner,
    drop_condition,
)


def test_class_conditioner_shape():
    conditioner = ClassConditioner(
        num_classes=3,
        embedding_dim=256,
    )

    class_ids = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    embeddings = conditioner(
        class_ids
    )

    assert embeddings.shape == (
        4,
        256,
    )


def test_null_class_forward():
    conditioner = ClassConditioner(
        num_classes=3,
        embedding_dim=64,
    )

    class_ids = torch.tensor(
        [0, 3],
        dtype=torch.long,
    )

    embeddings = conditioner(
        class_ids
    )

    assert embeddings.shape == (
        2,
        64,
    )

    assert torch.isfinite(
        embeddings
    ).all()


def test_real_classes_have_different_embeddings():
    conditioner = ClassConditioner(
        num_classes=3,
        embedding_dim=32,
    )

    class_ids = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    embeddings = conditioner(
        class_ids
    )

    assert not torch.equal(
        embeddings[0],
        embeddings[1],
    )

    assert not torch.equal(
        embeddings[1],
        embeddings[2],
    )


def test_class_id_out_of_range():
    conditioner = ClassConditioner(
        num_classes=3,
        embedding_dim=32,
    )

    class_ids = torch.tensor(
        [4],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError
    ):
        conditioner(
            class_ids
        )


def test_negative_class_id_rejected():
    conditioner = ClassConditioner(
        num_classes=3,
        embedding_dim=32,
    )

    class_ids = torch.tensor(
        [-1],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError
    ):
        conditioner(
            class_ids
        )


def test_drop_condition_probability_zero():
    class_ids = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    result = drop_condition(
        class_ids=class_ids,
        probability=0.0,
        null_class_id=3,
    )

    assert torch.equal(
        result,
        class_ids,
    )


def test_drop_condition_probability_one():
    class_ids = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    result = drop_condition(
        class_ids=class_ids,
        probability=1.0,
        null_class_id=3,
    )

    expected = torch.tensor(
        [3, 3, 3, 3],
        dtype=torch.long,
    )

    assert torch.equal(
        result,
        expected,
    )


def test_drop_condition_is_deterministic_with_generator():
    class_ids = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1],
        dtype=torch.long,
    )

    generator_a = torch.Generator()
    generator_a.manual_seed(
        123
    )

    generator_b = torch.Generator()
    generator_b.manual_seed(
        123
    )

    result_a = drop_condition(
        class_ids=class_ids,
        probability=0.5,
        null_class_id=3,
        generator=generator_a,
    )

    result_b = drop_condition(
        class_ids=class_ids,
        probability=0.5,
        null_class_id=3,
        generator=generator_b,
    )

    assert torch.equal(
        result_a,
        result_b,
    )


def test_gradient_reaches_class_embedding():
    conditioner = ClassConditioner(
        num_classes=3,
        embedding_dim=16,
    )

    class_ids = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    embeddings = conditioner(
        class_ids
    )

    loss = (
        embeddings ** 2
    ).mean()

    loss.backward()

    gradient = (
        conditioner
        .embedding
        .weight
        .grad
    )

    assert gradient is not None

    assert torch.isfinite(
        gradient
    ).all()

    assert gradient.abs().sum() > 0


def test_invalid_dropout_probability():
    class_ids = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError
    ):
        drop_condition(
            class_ids=class_ids,
            probability=1.1,
            null_class_id=3,
        )