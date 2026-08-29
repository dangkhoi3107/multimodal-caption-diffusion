from __future__ import annotations

from pathlib import Path

from src.data.product_dataset import (
    ProductImageDataset,
)
from src.text.tokenizer import (
    encode,
    padding_mask,
)
from src.text.vocabulary import (
    Vocabulary,
)


class TextProductDataset(
    ProductImageDataset
):
    """Phase 3 product dataset.

    Returns:
        image:        [3, H, W]
        class_id:     scalar long
        class_name:   str
        caption:      str
        token_ids:    [L] long
        padding_mask: [L] bool
    """

    def __init__(
        self,
        metadata_path: Path,
        vocabulary: Vocabulary,
        max_length: int,
    ) -> None:
        super().__init__(
            metadata_path=metadata_path
        )

        if max_length < 2:
            raise ValueError(
                "max_length must be at least 2"
            )

        self.vocabulary = vocabulary
        self.max_length = max_length

        for index, record in enumerate(
            self.records
        ):
            if "caption" not in record:
                raise ValueError(
                    f"Missing caption in "
                    f"record {index}"
                )

            caption = str(
                record["caption"]
            ).strip()

            if not caption:
                raise ValueError(
                    f"Empty caption in "
                    f"record {index}"
                )

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        item = super().__getitem__(
            index
        )

        record = self.records[
            index
        ]

        caption = str(
            record["caption"]
        )

        token_ids = encode(
            text=caption,
            vocabulary=self.vocabulary,
            max_length=self.max_length,
        )

        mask = padding_mask(
            token_ids=token_ids,
            vocabulary=self.vocabulary,
        )

        item[
            "caption"
        ] = caption

        item[
            "token_ids"
        ] = token_ids

        item[
            "padding_mask"
        ] = mask

        return item