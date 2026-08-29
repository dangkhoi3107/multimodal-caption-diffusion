import torch

from src.text.tokenizer import (
    decode,
    encode,
    normalize_text,
    padding_mask,
    tokenize,
)
from src.text.vocabulary import (
    build_vocabulary,
)


def make_vocab():
    captions = [
        "a red lifebuoy handwash pouch",
        "a blue dove deodorant tube",
        "a white dove body serum bottle",
    ]

    sequences = [
        tokenize(
            caption
        )
        for caption in captions
    ]

    return build_vocabulary(
        sequences
    )


def test_normalize_text():
    assert normalize_text(
        "  A   RED Pouch  "
    ) == "a red pouch"


def test_tokenize():
    assert tokenize(
        "A red pouch."
    ) == [
        "a",
        "red",
        "pouch",
    ]


def test_encode_shape_and_dtype():
    vocabulary = make_vocab()

    token_ids = encode(
        text=(
            "a red lifebuoy "
            "handwash pouch"
        ),
        vocabulary=vocabulary,
        max_length=10,
    )

    assert token_ids.shape == (
        10,
    )

    assert (
        token_ids.dtype
        == torch.long
    )


def test_encode_adds_bos_and_eos():
    vocabulary = make_vocab()

    token_ids = encode(
        text="a red pouch",
        vocabulary=vocabulary,
        max_length=8,
    )

    assert (
        token_ids[0].item()
        == vocabulary.bos_id
    )

    eos_positions = (
        token_ids
        == vocabulary.eos_id
    ).nonzero()

    assert (
        eos_positions.numel()
        > 0
    )


def test_unknown_token():
    vocabulary = make_vocab()

    token_ids = encode(
        text="a green pouch",
        vocabulary=vocabulary,
        max_length=8,
    )

    assert (
        vocabulary.unk_id
        in token_ids.tolist()
    )


def test_padding_mask():
    vocabulary = make_vocab()

    token_ids = encode(
        text="a red pouch",
        vocabulary=vocabulary,
        max_length=10,
    )

    mask = padding_mask(
        token_ids,
        vocabulary,
    )

    assert mask.dtype == torch.bool

    assert torch.equal(
        mask,
        (
            token_ids
            != vocabulary.pad_id
        ),
    )


def test_round_trip():
    vocabulary = make_vocab()

    original = (
        "a red lifebuoy "
        "handwash pouch"
    )

    token_ids = encode(
        text=original,
        vocabulary=vocabulary,
        max_length=10,
    )

    reconstructed = decode(
        token_ids,
        vocabulary,
    )

    assert (
        reconstructed
        == original
    )


def test_truncation_preserves_eos():
    vocabulary = make_vocab()

    token_ids = encode(
        text=(
            "a white dove body "
            "serum bottle"
        ),
        vocabulary=vocabulary,
        max_length=5,
    )

    assert (
        token_ids[-1].item()
        == vocabulary.eos_id
    )