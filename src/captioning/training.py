from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def caption_cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,L,V]")

    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [B,L]")

    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits and target_ids shapes do not match")

    valid = target_ids != pad_id
    count = int(valid.sum().item())

    if count <= 0:
        raise ValueError("batch has no non-PAD target tokens")

    vocab_size = logits.shape[-1]

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    ) / count

    if not torch.isfinite(loss):
        raise FloatingPointError("caption loss is NaN or Inf")

    return loss


@torch.no_grad()
def token_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> tuple[int, int]:
    predictions = logits.argmax(dim=-1)
    valid = target_ids != pad_id

    correct = int(
        ((predictions == target_ids) & valid)
        .sum()
        .item()
    )

    total = int(valid.sum().item())

    return correct, total


def gradients_are_finite(
    model: nn.Module,
) -> bool:
    found = False

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        found = True

        if not torch.isfinite(parameter.grad).all():
            return False

    return found


def train_caption_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    max_grad_norm: float = 1.0,
) -> dict:
    model.train()

    loss_sum = 0.0
    token_correct = 0
    token_total = 0
    sample_total = 0

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            images=images,
            input_ids=input_ids,
            padding_mask=padding_mask,
        )

        loss = caption_cross_entropy_loss(
            logits=logits,
            target_ids=target_ids,
            pad_id=pad_id,
        )

        loss.backward()

        if not gradients_are_finite(model):
            raise FloatingPointError(
                "missing or non-finite gradients"
            )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )

        if not math.isfinite(float(gradient_norm.item())):
            raise FloatingPointError(
                "gradient norm is non-finite"
            )

        optimizer.step()

        batch_size = images.shape[0]
        correct, total = token_accuracy(
            logits=logits.detach(),
            target_ids=target_ids,
            pad_id=pad_id,
        )

        loss_sum += float(loss.detach().item()) * batch_size
        token_correct += correct
        token_total += total
        sample_total += batch_size

    return {
        "loss": loss_sum / sample_total,
        "token_accuracy": (
            token_correct / token_total
            if token_total > 0
            else 0.0
        ),
    }


@torch.no_grad()
def validate_caption_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    pad_id: int,
) -> dict:
    model.eval()

    loss_sum = 0.0
    token_correct = 0
    token_total = 0
    sample_total = 0

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        logits = model(
            images=images,
            input_ids=input_ids,
            padding_mask=padding_mask,
        )

        loss = caption_cross_entropy_loss(
            logits=logits,
            target_ids=target_ids,
            pad_id=pad_id,
        )

        batch_size = images.shape[0]
        correct, total = token_accuracy(
            logits=logits,
            target_ids=target_ids,
            pad_id=pad_id,
        )

        loss_sum += float(loss.item()) * batch_size
        token_correct += correct
        token_total += total
        sample_total += batch_size

    return {
        "loss": loss_sum / sample_total,
        "token_accuracy": (
            token_correct / token_total
            if token_total > 0
            else 0.0
        ),
    }
