from __future__ import annotations

from scripts.evaluate_phase5_visual_conditioning import (
    build_class_mismatched_permutation,
)


def test_mismatched_permutation_is_one_to_one_and_cross_class():
    class_ids = (
        [0] * 4
        + [1] * 5
        + [2] * 3
    )

    permutation = (
        build_class_mismatched_permutation(
            class_ids
        )
    )

    assert sorted(
        permutation
    ) == list(
        range(
            len(
                class_ids
            )
        )
    )

    for (
        target_index,
        source_index,
    ) in enumerate(
        permutation
    ):
        assert (
            class_ids[
                target_index
            ]
            != class_ids[
                source_index
            ]
        )


def test_mismatched_permutation_is_deterministic():
    class_ids = (
        [0] * 4
        + [1] * 5
        + [2] * 3
    )

    first = (
        build_class_mismatched_permutation(
            class_ids
        )
    )

    second = (
        build_class_mismatched_permutation(
            class_ids
        )
    )

    assert (
        first
        == second
    )


def test_mismatched_permutation_rejects_impossible_distribution():
    class_ids = (
        [0] * 7
        + [1] * 2
        + [2] * 1
    )

    try:
        build_class_mismatched_permutation(
            class_ids
        )
    except ValueError:
        return

    raise AssertionError(
        "distribution with majority > 50% should be rejected"
    )
