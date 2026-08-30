"""Interactive Phase 3 text-to-image DDPM page."""

from __future__ import annotations

import streamlit as st
import torch
from PIL import Image

from demo.inference import DIFFUSION_CKPT_PATH
from demo.portfolio_data import PROMPT_EXAMPLES
from demo.ui import image_to_png_bytes, load_diffusion_model


st.title("Text → image playground")
st.write(
    "Generate one `64×64` product crop with the verified text-conditioned DDPM. "
    "Use a known prompt first; custom words outside the 19-token vocabulary map "
    "to `UNK`."
)

with st.container(border=True):
    with st.form("text_to_image_form"):
        prompt = st.selectbox(
            "Prompt",
            options=PROMPT_EXAMPLES,
            index=2,
            accept_new_options=True,
            placeholder="Choose a known prompt or type a custom one",
            help="Known concepts: Dove, Lifebuoy, serum, deodorant, handwash, bottle, tube, pouch, white, blue, and red.",
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
            value=False,
            help="Off streams about 20 previews; on streams all 1,000 reverse states.",
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
    progress_bar = live_panel.progress(0, text="Loading the diffusion checkpoint…")
    step_slot = live_panel.empty()

    try:
        if prompt is None:
            raise ValueError("Choose or enter a prompt before generation")
        with st.spinner("Restoring the text-conditioned DDPM…"):
            diffusion = load_diffusion_model(str(DIFFUSION_CKPT_PATH))

        preview_interval = (
            1
            if preview_every_step
            else max(diffusion.scheduler.num_timesteps // 20, 1)
        )

        def update_denoising_preview(
            completed_steps: int,
            total_steps: int,
            timestep: int,
            preview: Image.Image,
        ) -> None:
            """Render one live DDPM state without retaining the reverse chain."""

            progress_text = (
                "Initial Gaussian noise"
                if completed_steps == 0
                else f"Denoising · step {completed_steps:,}/{total_steps:,}"
            )
            preview_slot.image(preview, caption=progress_text, width=320)
            progress_bar.progress(completed_steps / total_steps, text=progress_text)
            step_slot.caption(
                f"Reverse timestep `t={timestep}` · preview every "
                f"{preview_interval} step(s)"
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
        progress_bar.progress(1.0, text="Denoising complete")
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
                f"**Preview interval:** `{diffusion_result['preview_interval']}`  \n"
                "**Reverse steps:** `1,000`  \n"
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
                file_name=f"ddpm_seed_{diffusion_result['seed']}.png",
                mime="image/png",
                on_click="ignore",
                icon=":material/download:",
                width="stretch",
            )

if not torch.cuda.is_available():
    st.warning(
        "CPU mode is available, but the verified 1,000-step reverse chain can take "
        "several minutes. The image-captioning page is much faster.",
        icon=":material/speed:",
    )

with st.expander("How this inference path works", icon=":material/schema:"):
    st.markdown(
        "`prompt → token IDs [1,L] → scratch text encoder → pooled condition → "
        "conditional U-Net ε prediction → classifier-free guidance → 1,000 DDPM "
        "reverse steps → image [1,3,64,64]`"
    )
