from pathlib import Path

from src.text.vocabulary import (
    build_vocabulary,
    Vocabulary,
)


def test_special_token_ids_are_fixed():
    vocabulary = build_vocabulary(
        [
            ["a", "red", "pouch"],
            ["a", "blue", "tube"],
        ]
    )

    assert vocabulary.pad_id == 0
    assert vocabulary.bos_id == 1
    assert vocabulary.eos_id == 2
    assert vocabulary.unk_id == 3


def test_vocabulary_is_deterministic():
    sequences_a = [
        ["red", "pouch"],
        ["blue", "tube"],
    ]

    sequences_b = [
        ["blue", "tube"],
        ["red", "pouch"],
    ]

    vocab_a = build_vocabulary(
        sequences_a
    )

    vocab_b = build_vocabulary(
        sequences_b
    )

    assert (
        vocab_a.token_to_id
        == vocab_b.token_to_id
    )


def test_min_frequency():
    vocabulary = build_vocabulary(
        [
            ["dove", "blue"],
            ["dove", "white"],
        ],
        min_frequency=2,
    )

    assert "dove" in (
        vocabulary.token_to_id
    )

    assert "blue" not in (
        vocabulary.token_to_id
    )

    assert "white" not in (
        vocabulary.token_to_id
    )


def test_unknown_token_maps_to_unk():
    vocabulary = build_vocabulary(
        [
            ["red", "pouch"],
        ]
    )

    assert (
        vocabulary.token_id(
            "unknown"
        )
        == vocabulary.unk_id
    )


def test_save_load_round_trip(
    tmp_path: Path,
):
    vocabulary = build_vocabulary(
        [
            ["dove", "serum"],
            ["lifebuoy", "pouch"],
        ]
    )

    path = (
        tmp_path
        / "vocabulary.json"
    )

    vocabulary.save(
        path
    )

    loaded = Vocabulary.load(
        path
    )

    assert (
        loaded.token_to_id
        == vocabulary.token_to_id
    )

    assert (
        loaded.id_to_token
        == vocabulary.id_to_token
    )