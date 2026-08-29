from __future__ import annotations

import torch

from src.captioning.model import CaptionModel


@torch.no_grad()
def greedy_generate(
    model: CaptionModel,
    images: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
) -> torch.Tensor:
    """Greedy autoregressive caption generation.

    Returns:
        LongTensor[B, <=max_length]
        Sequence includes BOS. Generation stops when every sample has
        emitted EOS or max_length is reached.
    """

    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    if max_length > model.max_length:
        raise ValueError(
            "generation max_length exceeds decoder max_length"
        )

    model.eval()

    image_tokens = model.encode_images(images)
    batch_size = images.shape[0]

    generated = torch.full(
        (batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=images.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=images.device,
    )

    for _ in range(max_length - 1):
        padding_mask = generated != pad_id

        logits = model.decoder(
            input_ids=generated,
            padding_mask=padding_mask,
            image_tokens=image_tokens,
        )

        next_token = logits[:, -1].argmax(dim=-1)

        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_id),
            next_token,
        )

        generated = torch.cat(
            [
                generated,
                next_token[:, None],
            ],
            dim=1,
        )

        finished = (
            finished
            | (next_token == eos_id)
        )

        if finished.all():
            break

    return generated
