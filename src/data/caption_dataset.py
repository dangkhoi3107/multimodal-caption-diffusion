from __future__ import annotations

from pathlib import Path

import torch

from src.data.text_product_dataset import TextProductDataset
from src.text.vocabulary import Vocabulary


class CaptionDataset(TextProductDataset):
    """Phase 4 image-captioning dataset.

    Reuses the Phase 3 processed product images, controlled captions,
    tokenizer, and train-only vocabulary.

    Contract for one sample:
        image:          FloatTensor[3, H, W], normalized to [-1, 1]
        class_id:       scalar LongTensor
        class_name:     str
        file_name:      str
        caption:        str
        input_ids:      LongTensor[L]
        target_ids:     LongTensor[L]
        padding_mask:   BoolTensor[L], True for real decoder input tokens
        target_mask:    BoolTensor[L], True for non-PAD target tokens

    Teacher-forcing shift:
        full_ids   = [BOS, w1, w2, ..., EOS, PAD, ...]
        input_ids  = full_ids[:-1]
        target_ids = full_ids[1:]

    `sequence_length` is the decoder sequence length L. Internally, one
    extra token is encoded so input_ids and target_ids both have length L.
    """

    def __init__(
        self,
        metadata_path: Path,
        vocabulary: Vocabulary,
        sequence_length: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2"
            )

        self.sequence_length = int(
            sequence_length
        )

        super().__init__(
            metadata_path=metadata_path,
            vocabulary=vocabulary,
            max_length=(
                self.sequence_length + 1
            ),
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        item = super().__getitem__(
            index
        )

        full_ids = item.pop(
            "token_ids"
        )

        # The parent dataset encoded one extra position.
        expected_full_shape = (
            self.sequence_length + 1,
        )

        if full_ids.shape != (
            expected_full_shape
        ):
            raise ValueError(
                "Unexpected encoded caption shape: "
                f"expected {expected_full_shape}, "
                f"got {tuple(full_ids.shape)}"
            )

        input_ids = (
            full_ids[:-1]
            .clone()
        )

        target_ids = (
            full_ids[1:]
            .clone()
        )

        decoder_padding_mask = (
            input_ids
            != self.vocabulary.pad_id
        )

        target_mask = (
            target_ids
            != self.vocabulary.pad_id
        )

        if input_ids.shape != (
            self.sequence_length,
        ):
            raise ValueError(
                "input_ids has wrong shape"
            )

        if target_ids.shape != (
            self.sequence_length,
        ):
            raise ValueError(
                "target_ids has wrong shape"
            )

        if decoder_padding_mask.dtype != torch.bool:
            raise TypeError(
                "padding_mask must be bool"
            )

        if target_mask.dtype != torch.bool:
            raise TypeError(
                "target_mask must be bool"
            )

        item[
            "input_ids"
        ] = input_ids

        item[
            "target_ids"
        ] = target_ids

        # Phase 4 meaning:
        # True = a real decoder input token.
        item[
            "padding_mask"
        ] = decoder_padding_mask

        item[
            "target_mask"
        ] = target_mask

        return item
