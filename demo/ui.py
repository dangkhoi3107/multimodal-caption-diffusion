"""Shared Streamlit UI helpers for the portfolio demo.

The module owns lightweight presentation helpers and cached demo-adapter
construction. Core preprocessing and model inference remain in
``demo.inference`` and ``src/``.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from demo.inference import CaptioningDemo, DiffusionDemo
from demo.portfolio_data import PRODUCT_EXAMPLES


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize one PIL image as PNG bytes for session state or download."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_resource(show_spinner=False, max_entries=2)
def load_caption_model(checkpoint_path: str) -> CaptioningDemo:
    """Restore and cache the Phase 4 captioning inference adapter."""

    return CaptioningDemo(checkpoint_path=Path(checkpoint_path))


@st.cache_resource(show_spinner=False, max_entries=2)
def load_diffusion_model(checkpoint_path: str) -> DiffusionDemo:
    """Restore and cache the Phase 3 diffusion inference adapter."""

    return DiffusionDemo(checkpoint_path=Path(checkpoint_path))


def checkpoint_status(path: Path) -> str:
    """Return a concise local checkpoint status without loading its tensors."""

    if not path.is_file():
        return "Downloads on first use"
    return f"Ready · {path.stat().st_size / (1024 ** 2):.1f} MB"


def render_product_gallery(*, show_split_counts: bool) -> None:
    """Render one fixed card per class from the bundled held-out crops."""

    columns = st.columns(3, gap="medium")
    for column, example in zip(columns, PRODUCT_EXAMPLES, strict=True):
        with column.container(border=True, height="stretch"):
            st.image(
                example.asset_path,
                caption=f"Class {example.class_id} · held-out test crop",
                width="stretch",
            )
            st.markdown(f"**{example.label}**")
            st.caption(example.description)
            if show_split_counts:
                st.caption(
                    f"{example.train_count} train · {example.valid_count} valid · "
                    f"{example.test_count} test"
                )


def render_footer() -> None:
    """Render the shared portfolio provenance footer."""

    st.caption(
        "Educational portfolio demo · PyTorch primitives · random initialization · "
        "no pretrained weights · reproducible artifacts in the repository"
    )
