from __future__ import annotations

from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from demo.portfolio_data import (
    PRODUCT_EXAMPLES,
    TEST_INSTANCES,
    TOTAL_DATASET_INSTANCES,
    TRAIN_INSTANCES,
    VALID_INSTANCES,
    get_product_example,
)


def test_streamlit_app_renders_without_loading_models() -> None:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "demo"
        / "streamlit_app.py"
    )

    app = AppTest.from_file(str(app_path))
    app.run(timeout=30)

    assert not app.exception
    assert [title.value for title in app.title] == [
        "Multimodal generation from scratch"
    ]
    assert [metric.label for metric in app.metric] == [
        "Processed crops",
        "Product classes",
        "Caption vocabulary",
        "Verified test suite",
    ]

    app.switch_page("app_pages/text_to_image.py").run(timeout=30)
    assert not app.exception
    assert [title.value for title in app.title] == ["Text → image playground"]
    preview_checkbox = next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Preview every denoising step"
    )
    assert preview_checkbox.value is False

    app.switch_page("app_pages/image_to_caption.py").run(timeout=30)
    assert not app.exception
    assert [title.value for title in app.title] == ["Image → caption playground"]
    assert len(app.image) >= 4

    app.switch_page("app_pages/evidence.py").run(timeout=30)
    assert not app.exception
    assert [title.value for title in app.title] == ["Evaluation evidence"]


def test_portfolio_examples_are_small_valid_held_out_assets() -> None:
    assert len(PRODUCT_EXAMPLES) == 3
    assert TOTAL_DATASET_INSTANCES == 683
    assert (TRAIN_INSTANCES, VALID_INSTANCES, TEST_INSTANCES) == (456, 147, 80)
    assert {example.class_id for example in PRODUCT_EXAMPLES} == {0, 1, 2}

    for example in PRODUCT_EXAMPLES:
        assert get_product_example(example.label) == example
        assert example.asset_path.is_file()
        assert "data/raw" not in example.asset_path.as_posix()
        with Image.open(example.asset_path) as image:
            assert image.mode == "RGB"
            assert image.size == (64, 64)
