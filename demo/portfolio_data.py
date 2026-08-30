"""Version-controlled portfolio metadata and curated demo assets.

This module describes the three supported product classes and points to a
small, reviewable set of held-out crops. It does not read the ignored raw or
processed data directories and it never restores a model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_ASSET_ROOT = Path(__file__).resolve().parent / "assets"
WORKFLOW_ASSET_PATH = REPOSITORY_ROOT / "assets" / "full-workflow.png"
CFG_GRID_ASSET_PATH = DEMO_ASSET_ROOT / "phase2_cfg_class_grid.png"


@dataclass(frozen=True)
class ProductExample:
    """Describe one class and one curated held-out image.

    ``asset_path`` resolves to one RGB crop used only for portfolio display and
    optional caption inference. Split counts describe the full processed
    dataset, not the number of bundled demo images.
    """

    class_id: int
    slug: str
    label: str
    description: str
    prompt: str
    reference_caption: str
    asset_filename: str
    train_count: int
    valid_count: int
    test_count: int

    @property
    def asset_path(self) -> Path:
        """Return the repository-local path to the curated PNG crop."""

        return DEMO_ASSET_ROOT / self.asset_filename

    @property
    def total_count(self) -> int:
        """Return this class's number of processed instances across splits."""

        return self.train_count + self.valid_count + self.test_count


PRODUCT_EXAMPLES = (
    ProductExample(
        class_id=0,
        slug="dove-body-serum",
        label="Dove body serum",
        description="White pump bottle · body serum",
        prompt="a white dove body serum bottle",
        reference_caption="a white dove body serum bottle",
        asset_filename="sample_dove_body_serum.png",
        train_count=156,
        valid_count=54,
        test_count=28,
    ),
    ProductExample(
        class_id=1,
        slug="dove-deodorant",
        label="Dove deodorant",
        description="Blue and white tube · deodorant",
        prompt="a blue dove deodorant tube",
        reference_caption="a dove deodorant in a blue tube",
        asset_filename="sample_dove_deodorant.png",
        train_count=203,
        valid_count=62,
        test_count=36,
    ),
    ProductExample(
        class_id=2,
        slug="lifebuoy-handwash",
        label="Lifebuoy handwash",
        description="Red refill pouch · handwash",
        prompt="a red lifebuoy handwash pouch",
        reference_caption="a lifebuoy handwash in a red pouch",
        asset_filename="sample_lifebuoy_handwash.png",
        train_count=97,
        valid_count=31,
        test_count=16,
    ),
)

PROMPT_EXAMPLES = tuple(example.prompt for example in PRODUCT_EXAMPLES)
TOTAL_DATASET_INSTANCES = sum(example.total_count for example in PRODUCT_EXAMPLES)
TRAIN_INSTANCES = sum(example.train_count for example in PRODUCT_EXAMPLES)
VALID_INSTANCES = sum(example.valid_count for example in PRODUCT_EXAMPLES)
TEST_INSTANCES = sum(example.test_count for example in PRODUCT_EXAMPLES)


def get_product_example(label: str) -> ProductExample:
    """Return the curated product whose display label matches ``label``.

    Raises:
        ValueError: If ``label`` is not one of the three supported classes.
    """

    for example in PRODUCT_EXAMPLES:
        if example.label == label:
            return example
    supported = ", ".join(example.label for example in PRODUCT_EXAMPLES)
    raise ValueError(f"Unknown product label {label!r}; expected one of: {supported}")
