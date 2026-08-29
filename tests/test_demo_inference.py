from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from demo.inference import (
    ensure_checkpoint_available,
    load_checkpoint,
    preprocess_product_image,
    restore_vocabulary,
    tensor_to_pil,
)
from src.text.vocabulary import SPECIAL_TOKENS


def test_preprocess_product_image_contract() -> None:
    image = Image.new(
        "RGB",
        (4, 2),
        color=(255, 0, 0),
    )

    tensor = preprocess_product_image(
        image=image,
        image_size=8,
    )

    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0

    # The 2:1 image becomes 8x4 with two white letterbox rows per side.
    assert torch.equal(tensor[0, :, 0, 0], torch.ones(3))
    assert torch.equal(
        tensor[0, :, 3, 3],
        torch.tensor([1.0, -1.0, -1.0]),
    )


def test_preprocess_rejects_invalid_boundary_inputs() -> None:
    with pytest.raises(TypeError, match="PIL"):
        preprocess_product_image("not an image", 64)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        preprocess_product_image(Image.new("RGB", (1, 1)), 0)


def test_tensor_to_pil_contract() -> None:
    tensor = torch.zeros(3, 5, 7)

    image = tensor_to_pil(tensor)

    assert image.mode == "RGB"
    assert image.size == (7, 5)
    assert image.getpixel((0, 0)) == (128, 128, 128)


def test_tensor_to_pil_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\[3, H, W\]"):
        tensor_to_pil(torch.zeros(1, 3, 8, 8))


def test_restore_vocabulary_round_trip() -> None:
    tokens = (*SPECIAL_TOKENS, "a", "bottle")
    serialized = {
        "token_to_id": {
            token: index
            for index, token in enumerate(tokens)
        },
        "id_to_token": list(tokens),
    }

    vocabulary = restore_vocabulary(serialized)

    assert vocabulary.id_to_token == tokens
    assert vocabulary.token_id("bottle") == 5


def test_restore_vocabulary_rejects_invalid_data() -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint vocabulary"):
        restore_vocabulary({"id_to_token": []})


def test_load_checkpoint_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_validates_required_keys(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"config": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="missing required keys"):
        load_checkpoint(checkpoint_path)


def test_ensure_checkpoint_downloads_and_verifies_file(tmp_path: Path) -> None:
    payload = b"project-trained-checkpoint"
    source = tmp_path / "source.pt"
    destination = tmp_path / "cache" / "model.pt"
    source.write_bytes(payload)

    resolved = ensure_checkpoint_available(
        path=destination,
        download_url=source.as_uri(),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert resolved == destination.resolve()
    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.part"))


def test_ensure_checkpoint_rejects_bad_checksum_and_cleans_temp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "cache" / "model.pt"
    source.write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure_checkpoint_available(
            path=destination,
            download_url=source.as_uri(),
            expected_sha256="0" * 64,
        )

    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))
