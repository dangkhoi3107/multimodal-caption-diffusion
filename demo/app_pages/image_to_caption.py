"""Interactive Phase 4 image-captioning page with curated example inputs."""

from __future__ import annotations

import streamlit as st
from PIL import Image, UnidentifiedImageError

from demo.inference import (
    CAPTION_CKPT_PATH,
    preprocess_product_image,
    tensor_to_pil,
)
from demo.portfolio_data import PRODUCT_EXAMPLES, get_product_example
from demo.ui import image_to_png_bytes, load_caption_model, render_product_gallery


st.title("Image → caption playground")
st.write(
    "Start with one bundled held-out crop or upload a centered product image. The "
    "same preprocessing and autoregressive decoding path used in evaluation runs "
    "here without teacher forcing."
)

render_product_gallery(show_split_counts=False)

source_mode = st.segmented_control(
    "Input source",
    options=("Built-in examples", "Upload an image"),
    default="Built-in examples",
    required=True,
    key="caption_source_mode",
    width="stretch",
)

source_image: Image.Image | None = None
source_label = ""
reference_caption: str | None = None

input_column, settings_column = st.columns((1, 1), gap="large")
with input_column:
    with st.container(border=True, height="stretch"):
        if source_mode == "Built-in examples":
            selected_label = st.selectbox(
                "Product example",
                options=tuple(example.label for example in PRODUCT_EXAMPLES),
                index=0,
                key="caption_product_example",
            )
            selected_example = get_product_example(selected_label)
            source_image = Image.open(selected_example.asset_path).convert("RGB")
            source_label = f"{selected_example.label} · held-out test crop"
            reference_caption = selected_example.reference_caption
            st.image(source_image, caption=source_label, width="stretch")
        else:
            uploaded_file = st.file_uploader(
                "Product image",
                type=("png", "jpg", "jpeg"),
                max_upload_size=10,
                help="Maximum file size: 10 MB.",
                key="caption_upload",
            )
            if uploaded_file is not None:
                try:
                    source_image = Image.open(uploaded_file).convert("RGB")
                    source_label = "Uploaded image"
                    st.image(source_image, caption=source_label, width="stretch")
                except (UnidentifiedImageError, OSError) as error:
                    st.error(
                        f"The uploaded file is not a readable image: {error}",
                        icon=":material/error:",
                    )

with settings_column:
    with st.container(border=True, height="stretch"):
        st.markdown("**Decoding settings**")
        strategy_label = st.segmented_control(
            "Strategy",
            options=("Greedy", "Beam search"),
            default="Greedy",
            required=True,
            key="caption_strategy",
            width="stretch",
        )
        with st.form("image_to_caption_form", border=False):
            beam_size = st.slider(
                "Beam size",
                min_value=2,
                max_value=5,
                value=3,
                disabled=strategy_label != "Beam search",
            )
            length_penalty = st.slider(
                "Length penalty",
                min_value=0.0,
                max_value=1.5,
                value=0.6,
                step=0.1,
                disabled=strategy_label != "Beam search",
            )
            generate_caption = st.form_submit_button(
                "Generate caption",
                type="primary",
                icon=":material/subtitles:",
                width="stretch",
            )
        st.caption(
            "Greedy chooses the best next token. Beam search retains several "
            "candidate sequences and applies a length penalty."
        )

caption_slot = st.container()

if generate_caption:
    if source_image is None:
        caption_slot.warning(
            "Choose a built-in example or upload a readable image first.",
            icon=":material/upload:",
        )
    else:
        try:
            model_input = tensor_to_pil(preprocess_product_image(source_image, 64)[0])
            strategy = "beam" if strategy_label == "Beam search" else "greedy"

            with caption_slot.skeleton(height=320):
                captioner = load_caption_model(str(CAPTION_CKPT_PATH))
                caption = captioner.generate_caption(
                    image=source_image,
                    strategy=strategy,
                    beam_size=int(beam_size),
                    length_penalty=float(length_penalty),
                )
                st.session_state["caption_result"] = {
                    "source": image_to_png_bytes(source_image),
                    "source_label": source_label,
                    "model_input": image_to_png_bytes(model_input),
                    "caption": caption,
                    "reference_caption": reference_caption,
                    "strategy": strategy_label,
                    "parameter_count": captioner.parameter_count,
                }
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
            caption_slot.error(str(error), icon=":material/error:")

caption_result = st.session_state.get("caption_result")
if caption_result:
    with caption_slot.container(border=True):
        source_column, model_column = st.columns(2)
        with source_column:
            st.image(
                caption_result["source"],
                caption=caption_result["source_label"],
                width="stretch",
            )
        with model_column:
            st.image(
                caption_result["model_input"],
                caption="Model input · 64×64 letterbox",
                width="stretch",
            )

        st.subheader("Generated caption")
        st.success(caption_result["caption"], icon=":material/subtitles:")
        if caption_result["reference_caption"] is not None:
            st.caption(
                f"Held-out reference: “{caption_result['reference_caption']}”"
            )
        st.caption(
            f"{caption_result['strategy']} · "
            f"{caption_result['parameter_count']:,} learned parameters"
        )

with st.expander("How this inference path works", icon=":material/schema:"):
    st.markdown(
        "`PIL image → letterbox → image [1,3,64,64] → scratch CNN → visual "
        "tokens [1,64,256] → causal self-attention + visual cross-attention → "
        "token logits [1,L,19] → greedy/beam decoding → caption`"
    )
