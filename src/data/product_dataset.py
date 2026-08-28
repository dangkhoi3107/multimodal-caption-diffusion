import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ProductImageDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
    ) -> None:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        self.metadata_path = metadata_path
        self.root = metadata_path.parent

        self.records: list[dict] = []

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                required_fields = [
                    "file_name",
                    "class_id",
                    "class_name",
                    "image_size",
                ]

                for field in required_fields:
                    if field not in record:
                        raise ValueError(
                            f"Missing field '{field}' "
                            f"at line {line_number}"
                        )

                self.records.append(record)

        if not self.records:
            raise ValueError(
                "Metadata contains no records"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        record = self.records[index]

        image_path = (
            self.root / record["file_name"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Processed image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            array = np.asarray(
                image,
                dtype=np.uint8,
            ).copy()

        expected_size = int(
            record["image_size"]
        )

        expected_shape = (
            expected_size,
            expected_size,
            3,
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"Expected image shape "
                f"{expected_shape}, got {array.shape}"
            )

        tensor = torch.from_numpy(
            array
        ).permute(
            2,
            0,
            1,
        )

        tensor = tensor.float()

        tensor = (
            tensor / 127.5
        ) - 1.0

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                "Image tensor contains non-finite values"
            )

        return {
            "image": tensor,
            "class_id": torch.tensor(
                int(record["class_id"]),
                dtype=torch.long,
            ),
            "class_name": str(
                record["class_name"]
            ),
            "file_name": str(
                record["file_name"]
            ),
        }