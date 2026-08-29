from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = (
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
)


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def token_id(
        self,
        token: str,
    ) -> int:
        return self.token_to_id.get(
            token,
            self.unk_id,
        )

    def token(
        self,
        token_id: int,
    ) -> str:
        if (
            token_id < 0
            or token_id >= len(self)
        ):
            raise ValueError(
                f"token_id out of range: "
                f"{token_id}"
            )

        return self.id_to_token[
            token_id
        ]

    def to_dict(self) -> dict:
        return {
            "token_to_id": (
                self.token_to_id
            ),
            "id_to_token": list(
                self.id_to_token
            ),
        }

    def save(
        self,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "Vocabulary":
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        token_to_id = {
            str(token): int(token_id)
            for token, token_id
            in data[
                "token_to_id"
            ].items()
        }

        id_to_token = tuple(
            str(token)
            for token
            in data[
                "id_to_token"
            ]
        )

        vocabulary = cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )

        vocabulary.validate()

        return vocabulary

    def validate(self) -> None:
        if len(
            self.token_to_id
        ) != len(
            self.id_to_token
        ):
            raise ValueError(
                "Vocabulary mappings "
                "have different sizes"
            )

        for index, token in enumerate(
            self.id_to_token
        ):
            mapped = (
                self.token_to_id.get(
                    token
                )
            )

            if mapped != index:
                raise ValueError(
                    "token_to_id and "
                    "id_to_token mismatch"
                )

        for expected_id, token in enumerate(
            SPECIAL_TOKENS
        ):
            actual_id = (
                self.token_to_id.get(
                    token
                )
            )

            if actual_id != expected_id:
                raise ValueError(
                    f"{token} must have "
                    f"ID {expected_id}"
                )


def build_vocabulary(
    token_sequences: Iterable[
        Iterable[str]
    ],
    min_frequency: int = 1,
) -> Vocabulary:
    if min_frequency <= 0:
        raise ValueError(
            "min_frequency must be positive"
        )

    counter: Counter[str] = Counter()

    for sequence in token_sequences:
        counter.update(
            sequence
        )

    normal_tokens = sorted(
        token
        for token, frequency
        in counter.items()
        if frequency >= min_frequency
        and token not in SPECIAL_TOKENS
    )

    id_to_token = (
        *SPECIAL_TOKENS,
        *normal_tokens,
    )

    token_to_id = {
        token: index
        for index, token
        in enumerate(
            id_to_token
        )
    }

    vocabulary = Vocabulary(
        token_to_id=token_to_id,
        id_to_token=tuple(
            id_to_token
        ),
    )

    vocabulary.validate()

    return vocabulary