"""Streamlit router for the multimodal generation portfolio."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st
import torch


# ``streamlit run demo/streamlit_app.py`` adds ``demo/`` rather than the
# repository root to sys.path. Resolve the project root before package imports.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo.inference import CAPTION_CKPT_PATH, DIFFUSION_CKPT_PATH
from demo.ui import checkpoint_status, render_footer


st.set_page_config(
    page_title="Multimodal generation from scratch",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("diffusion_result", None)
st.session_state.setdefault("caption_result", None)

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/text_to_image.py",
            title="Text to image",
            icon=":material/auto_awesome:",
        ),
        st.Page(
            "app_pages/image_to_caption.py",
            title="Image to caption",
            icon=":material/subtitles:",
        ),
        st.Page(
            "app_pages/evidence.py",
            title="Evidence",
            icon=":material/analytics:",
        ),
    ],
    position="top",
)

with st.sidebar:
    st.header("Project snapshot")
    st.caption(
        "Two compact generative models trained from random initialization on "
        "one three-class product domain."
    )

    with st.container(border=True):
        st.markdown("**Text → image checkpoint**")
        st.caption(checkpoint_status(DIFFUSION_CKPT_PATH))
        st.markdown("**Image → caption checkpoint**")
        st.caption(checkpoint_status(CAPTION_CKPT_PATH))

    device_label = "CUDA" if torch.cuda.is_available() else "CPU"
    st.badge(
        f"Inference device: {device_label}",
        color="green" if torch.cuda.is_available() else "orange",
        icon=":material/memory:",
    )

    st.link_button(
        "View source on GitHub",
        "https://github.com/dangkhoi3107/multimodal-caption-diffusion",
        icon=":material/code:",
        width="stretch",
    )

    with st.expander("Honest scope", icon=":material/info:"):
        st.markdown(
            "- `64×64` RGB research outputs\n"
            "- Three product SKU classes\n"
            "- Controlled vocabulary of 19 tokens\n"
            "- Full 1,000-step DDPM sampler\n"
            "- Packaging text and logos are not expected to be legible"
        )

page.run()
render_footer()
