from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


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
    assert any(
        "Generate a product image" in header.value
        for header in app.header
    )
    preview_checkbox = next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Preview every denoising step"
    )
    assert preview_checkbox.value is True
