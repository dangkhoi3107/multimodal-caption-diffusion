"""Portfolio overview page: scope, dataset, architecture, and tensor flow."""

from __future__ import annotations

import streamlit as st

from demo.portfolio_data import (
    PRODUCT_EXAMPLES,
    TEST_INSTANCES,
    TOTAL_DATASET_INSTANCES,
    TRAIN_INSTANCES,
    VALID_INSTANCES,
    WORKFLOW_ASSET_PATH,
)
from demo.ui import render_product_gallery


st.title("Multimodal generation from scratch")
st.write(
    "A bidirectional product-domain study built directly with PyTorch primitives: "
    "text-conditioned diffusion generates a product image, while a scratch CNN "
    "and Transformer generate a caption from an image."
)

with st.container(horizontal=True):
    st.badge("Phase 3 · text-conditioned DDPM", color="blue")
    st.badge("Phase 4 · image captioning", color="violet")
    st.badge("Phase 5 · integrated evaluation", color="green")

metric_columns = st.columns(4)
metric_columns[0].metric("Processed crops", f"{TOTAL_DATASET_INSTANCES}")
metric_columns[1].metric("Product classes", f"{len(PRODUCT_EXAMPLES)}")
metric_columns[2].metric("Caption vocabulary", "19 tokens")
metric_columns[3].metric("Verified test suite", "209 passed")

st.subheader("The 60-second project story")
story_columns = st.columns(3, gap="medium")
with story_columns[0].container(border=True, height="stretch"):
    st.markdown(":material/database: **1 · Prepare the product domain**")
    st.write(
        "COCO polygons become aspect-ratio-preserving `64×64` crops with stable "
        "class IDs and source traceability."
    )
with story_columns[1].container(border=True, height="stretch"):
    st.markdown(":material/auto_awesome: **2 · Learn text → image**")
    st.write(
        "A custom text encoder conditions a U-Net noise predictor. Classifier-free "
        "guidance steers the 1,000-step reverse DDPM chain."
    )
with story_columns[2].container(border=True, height="stretch"):
    st.markdown(":material/subtitles: **3 · Learn image → text**")
    st.write(
        "A scratch CNN emits 64 visual tokens. A causal Transformer decoder uses "
        "self-attention and visual cross-attention to generate captions."
    )

st.image(
    WORKFLOW_ASSET_PATH,
    caption="End-to-end research workflow and phase gates",
    width="stretch",
)

st.subheader("What the models actually see")
st.write(
    "The repository contains 683 processed instances from three SKU classes—not "
    "three training images. Only the three held-out crops below are bundled with "
    "the public demo so visitors can understand the domain without publishing the "
    "ignored raw dataset."
)
render_product_gallery(show_split_counts=True)
st.caption(
    f"Dataset split: {TRAIN_INSTANCES} train · {VALID_INSTANCES} validation · "
    f"{TEST_INSTANCES} test. Curated demo crops come from the held-out test split. "
    "Source dataset: [Stock Segmentation on Roboflow](https://universe.roboflow.com/"
    "dfdfdfd-d1nyr/stock_segmentation-dzocz), CC BY 4.0."
)

st.subheader("Runtime contracts")
contract_rows = [
    {
        "Stage": "Product preprocessing",
        "Input": "PIL RGB image",
        "Output": "image [1, 3, 64, 64]",
        "Invariant": "float32 in [-1, 1]; aspect ratio preserved",
    },
    {
        "Stage": "Text-conditioned DDPM",
        "Input": "token IDs [1, L] + noise [1, 3, 64, 64]",
        "Output": "RGB image 64×64",
        "Invariant": "same prompt + CFG + seed is deterministic",
    },
    {
        "Stage": "Captioning encoder",
        "Input": "image [B, 3, 64, 64]",
        "Output": "visual tokens [B, 64, 256]",
        "Invariant": "all parameters train end to end",
    },
    {
        "Stage": "Captioning decoder",
        "Input": "tokens [B, L] + visual tokens",
        "Output": "logits [B, L, 19]",
        "Invariant": "causal mask prevents future-token leakage",
    },
]
st.table(contract_rows)

with st.expander("What “from scratch” means here", icon=":material/build:"):
    st.markdown(
        "The project implements its own noise schedule, forward/reverse DDPM, "
        "timestep embedding, U-Net blocks, classifier-free guidance, tokenizer, "
        "scaled dot-product attention, multi-head attention, CNN encoder, "
        "Transformer decoder, greedy decoding, beam search, BLEU, and training "
        "loops. It uses no pretrained weights, CLIP, Diffusers, Hugging Face "
        "Transformers, `torch.nn.MultiheadAttention`, or `torch.nn.Transformer`."
    )
