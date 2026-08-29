"""Streamlit entry point for the bidirectional portfolio playground."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

# ``streamlit run demo/streamlit_app.py`` adds ``demo/`` rather than the
# repository root to sys.path. Resolve the project root before package imports.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo.inference import (
    CAPTION_CKPT_PATH,
    DIFFUSION_CKPT_PATH,
    CaptioningDemo,
    DiffusionDemo,
    preprocess_product_image,
    tensor_to_pil,
)


PROMPT_EXAMPLES = (
    "a red lifebuoy handwash pouch",
    "a white dove body serum bottle",
    "a blue dove deodorant bottle",
)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize one PIL image as PNG bytes for display and download."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_resource(show_spinner=False, max_entries=2)
def load_caption_model(checkpoint_path: str) -> CaptioningDemo:
    """Restore and cache the Phase 4 captioning model."""

    return CaptioningDemo(checkpoint_path=Path(checkpoint_path))


@st.cache_resource(show_spinner=False, max_entries=2)
def load_diffusion_model(checkpoint_path: str) -> DiffusionDemo:
    """Restore and cache the Phase 3 text-conditioned diffusion model."""

    return DiffusionDemo(checkpoint_path=Path(checkpoint_path))


def checkpoint_status(path: Path) -> str:
    """Return a concise local checkpoint status string."""

    if not path.is_file():
        return "Downloads on first use"
    return f"Ready · {path.stat().st_size / (1024 ** 2):.1f} MB"


st.set_page_config(
    page_title="Multimodal generation playground",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("diffusion_result", None)
st.session_state.setdefault("caption_result", None)

with st.sidebar:
    st.header("Project snapshot")
    st.caption(
        "Two small generative models trained from random initialization. "
        "No pretrained backbone, CLIP, Diffusers, or high-level Transformer."
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

    st.subheader("What this demonstrates")
    st.markdown(
        "- DDPM schedule and 1,000-step reverse sampler\n"
        "- Text encoder and classifier-free guidance\n"
        "- CNN visual tokens and causal Transformer decoding\n"
        "- Greedy and beam-search caption generation"
    )

    with st.expander("Honest limitations", icon=":material/info:"):
        st.markdown(
            "- Images are intentionally low-resolution (`64×64`).\n"
            "- Training data covers three product SKUs.\n"
            "- Packaging text and logos are not expected to be legible.\n"
            "- Text → image is slow on CPU because the verified sampler uses all "
            "1,000 DDPM steps.\n"
            "- Captioning works best on a centered single-product crop."
        )

st.title("Multimodal generation from scratch")
st.write(
    "Explore both directions of the same product domain: describe a product to "
    "generate an image, or upload a product image to generate a caption."
)

with st.container(horizontal=True):
    st.badge("Phase 3 · Text-conditioned DDPM", color="blue")
    st.badge("Phase 4 · Image captioning", color="violet")
    st.badge("Phase 5 · Integrated demo", color="green")

mode = st.segmented_control(
    "Choose a playground",
    options=("Text → image", "Image → caption"),
    default="Text → image",
    required=True,
    key="playground_mode",
    width="stretch",
)

if mode == "Text → image":
    st.header("Generate a product image")
    st.caption(
        "The model vocabulary is intentionally small. Start with one of the "
        "examples below, then change brand, color, product, or package words. "
        "While sampling, the preview updates from Gaussian noise to the final image."
    )

    with st.container(border=True):
        st.markdown("**Prompt examples**")
        for example in PROMPT_EXAMPLES:
            st.code(example, language=None)

        with st.form("text_to_image_form"):
            prompt = st.text_input(
                "Prompt",
                value=PROMPT_EXAMPLES[0],
                max_chars=120,
                help="Unknown words map to the model's UNK token.",
            )
            guidance_scale = st.slider(
                "Classifier-free guidance",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="0 is unconditional; 1 is conditional; larger values emphasize the prompt.",
            )
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
            )
            preview_every_step = st.checkbox(
                "Preview every denoising step",
                value=True,
                help=(
                    "Streams all 1,000 reverse states. Disable this for a "
                    "lighter preview with about 20 updates."
                ),
            )
            generate_image = st.form_submit_button(
                "Generate image",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

    result_slot = st.container()

    if generate_image:
        live_panel = result_slot.container(border=True)
        live_panel.markdown("**Live denoising preview**")
        preview_slot = live_panel.empty()
        progress_bar = live_panel.progress(
            0,
            text="Loading the diffusion checkpoint…",
        )
        step_slot = live_panel.empty()

        try:
            with st.spinner("Downloading or restoring the text-conditioned DDPM…"):
                diffusion = load_diffusion_model(str(DIFFUSION_CKPT_PATH))

            preview_interval = (
                1
                if preview_every_step
                else max(
                    diffusion.scheduler.num_timesteps // 20,
                    1,
                )
            )

            def update_denoising_preview(
                completed_steps: int,
                total_steps: int,
                timestep: int,
                preview: Image.Image,
            ) -> None:
                """Render one live DDPM state without storing all 1,000 frames."""

                if completed_steps == 0:
                    progress_text = "Initial Gaussian noise"
                else:
                    progress_text = (
                        f"Denoising · step {completed_steps:,}/{total_steps:,}"
                    )

                preview_slot.image(
                    preview,
                    caption=progress_text,
                    width=320,
                )
                progress_bar.progress(
                    completed_steps / total_steps,
                    text=progress_text,
                )
                step_slot.caption(
                    f"Reverse timestep `t={timestep}` · "
                    + (
                        "previewing every step"
                        if preview_interval == 1
                        else f"preview every {preview_interval} steps"
                    )
                )

            unknown_tokens = diffusion.unknown_prompt_tokens(prompt)
            generated_image = diffusion.generate_image(
                prompt=prompt,
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                progress_callback=update_denoising_preview,
                preview_interval=preview_interval,
            )
            st.session_state["diffusion_result"] = {
                "image": image_to_png_bytes(generated_image),
                "prompt": prompt,
                "guidance_scale": float(guidance_scale),
                "seed": int(seed),
                "unknown_tokens": unknown_tokens,
                "parameter_count": diffusion.parameter_count,
                "preview_interval": preview_interval,
            }
            progress_bar.progress(
                1.0,
                text="Denoising complete",
            )
            live_panel.empty()
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
            live_panel.empty()
            result_slot.error(str(error), icon=":material/error:")

    diffusion_result = st.session_state.get("diffusion_result")
    if diffusion_result:
        with result_slot.container(border=True):
            left, right = st.columns((2, 1), vertical_alignment="center")
            with left:
                st.image(
                    diffusion_result["image"],
                    caption=diffusion_result["prompt"],
                    width="stretch",
                )
            with right:
                st.subheader("Generation details")
                st.markdown(
                    f"**Seed:** `{diffusion_result['seed']}`  \n"
                    f"**CFG:** `{diffusion_result['guidance_scale']:.1f}`  \n"
                    f"**Parameters:** `{diffusion_result['parameter_count']:,}`  \n"
                    f"**Preview:** `every {diffusion_result['preview_interval']} "
                    "step(s)`  \n"
                    "**Output:** `64×64 RGB`"
                )
                if diffusion_result["unknown_tokens"]:
                    unknown_text = ", ".join(diffusion_result["unknown_tokens"])
                    st.warning(
                        f"Mapped to UNK: {unknown_text}",
                        icon=":material/warning:",
                    )
                st.download_button(
                    "Download PNG",
                    data=diffusion_result["image"],
                    file_name=(
                        f"ddpm_seed_{diffusion_result['seed']}.png"
                    ),
                    mime="image/png",
                    on_click="ignore",
                    icon=":material/download:",
                    width="stretch",
                )

    if not torch.cuda.is_available():
        st.warning(
            "This environment uses CPU. Text-to-image runs the verified 1,000-step "
            "DDPM chain and can take several minutes. Captioning is much faster.",
            icon=":material/speed:",
        )

elif mode == "Image → caption":
    st.header("Caption a product image")
    st.caption(
        "Upload a PNG or JPEG containing one centered product. The demo preserves "
        "aspect ratio, adds a white letterbox, and normalizes pixels to [-1, 1]."
    )

    with st.container(border=True):
        with st.form("image_to_caption_form"):
            uploaded_file = st.file_uploader(
                "Product image",
                type=("png", "jpg", "jpeg"),
                max_upload_size=10,
                help="Maximum file size: 10 MB.",
            )
            strategy_label = st.segmented_control(
                "Decoding strategy",
                options=("Greedy", "Beam search"),
                default="Greedy",
                required=True,
                width="stretch",
            )
            beam_size = st.slider(
                "Beam size",
                min_value=2,
                max_value=5,
                value=3,
                help="Used only when the decoding strategy is Beam search.",
            )
            length_penalty = st.slider(
                "Length penalty",
                min_value=0.0,
                max_value=1.5,
                value=0.6,
                step=0.1,
                help="Used only when the decoding strategy is Beam search.",
            )
            generate_caption = st.form_submit_button(
                "Generate caption",
                type="primary",
                icon=":material/subtitles:",
                width="stretch",
            )

    caption_slot = st.container()

    if generate_caption:
        if uploaded_file is None:
            caption_slot.warning(
                "Upload a PNG or JPEG before generating a caption.",
                icon=":material/upload:",
            )
        else:
            try:
                source_image = Image.open(uploaded_file).convert("RGB")
                input_size = 64
                model_input = tensor_to_pil(
                    preprocess_product_image(source_image, input_size)[0]
                )
                strategy = "beam" if strategy_label == "Beam search" else "greedy"

                with caption_slot.skeleton(height=400):
                    captioner = load_caption_model(str(CAPTION_CKPT_PATH))
                    caption = captioner.generate_caption(
                        image=source_image,
                        strategy=strategy,
                        beam_size=int(beam_size),
                        length_penalty=float(length_penalty),
                    )
                    st.session_state["caption_result"] = {
                        "source": image_to_png_bytes(source_image),
                        "model_input": image_to_png_bytes(model_input),
                        "caption": caption,
                        "strategy": strategy_label,
                        "parameter_count": captioner.parameter_count,
                    }
            except (UnidentifiedImageError, OSError) as error:
                caption_slot.error(
                    f"The uploaded file is not a readable image: {error}",
                    icon=":material/error:",
                )
            except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
                caption_slot.error(str(error), icon=":material/error:")

    caption_result = st.session_state.get("caption_result")
    if caption_result:
        with caption_slot.container(border=True):
            source_column, model_column = st.columns(2)
            with source_column:
                st.image(
                    caption_result["source"],
                    caption="Uploaded image",
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
            st.caption(
                f"{caption_result['strategy']} · "
                f"{caption_result['parameter_count']:,} learned parameters"
            )

st.caption(
    "Educational portfolio demo · PyTorch primitives · random initialization · "
    "local checkpoints · see README.md for metrics and reproducibility commands"
)
