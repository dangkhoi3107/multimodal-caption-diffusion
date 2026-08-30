"""Phase 5 evidence page with evaluation results and honest limitations."""

from __future__ import annotations

import streamlit as st

from demo.portfolio_data import CFG_GRID_ASSET_PATH


st.title("Evaluation evidence")
st.write(
    "The demo is backed by held-out metrics, fixed-seed conditioning checks, "
    "visual-token ablations, and contract tests—not only hand-picked outputs."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Caption test samples", "80")
metric_columns[1].metric("Greedy BLEU-4", "0.4906")
metric_columns[2].metric("Beam BLEU-4", "0.5152", delta="+0.0246")
metric_columns[3].metric("Class accuracy", "100%")

st.subheader("Caption decoding comparison")
st.table(
    [
        {
            "Decoder": "Greedy",
            "Exact match": "31.25%",
            "BLEU-1": "0.8593",
            "BLEU-2": "0.7284",
            "BLEU-3": "0.5850",
            "BLEU-4": "0.4906",
        },
        {
            "Decoder": "Beam (k=3)",
            "Exact match": "30.00%",
            "BLEU-1": "0.8587",
            "BLEU-2": "0.7412",
            "BLEU-3": "0.6111",
            "BLEU-4": "0.5152",
        },
    ]
)
st.caption(
    "Local demo checkpoint: epoch 20, selected before test evaluation. Beam search "
    "improves BLEU-4 but slightly reduces exact match, so both are reported."
)

st.subheader("Does captioning actually use the image?")
ablation_columns = st.columns(3)
with ablation_columns[0].container(border=True, height="stretch"):
    st.metric("Real visual tokens", "0.4906 BLEU-4")
    st.caption("100% controlled class accuracy")
with ablation_columns[1].container(border=True, height="stretch"):
    st.metric("Zero visual tokens", "0.2385 BLEU-4", delta="-0.2521")
    st.caption("35% controlled class accuracy")
with ablation_columns[2].container(border=True, height="stretch"):
    st.metric("Mismatched class tokens", "0.0000 BLEU-4", delta="-0.4905")
    st.caption("0% controlled class accuracy")
st.write(
    "Performance collapses when visual tokens are removed or replaced by another "
    "class, which is direct evidence that the decoder conditions on the image."
)

st.subheader("Fixed-noise class conditioning")
st.image(
    CFG_GRID_ASSET_PATH,
    caption=(
        "Phase 2 classifier-free-guidance sanity check · same stochastic path, "
        "different class condition · seed 342"
    ),
    width="stretch",
)
st.caption(
    "The Phase 3 playground extends the same conditioning idea from class IDs to "
    "scratch text embeddings. These `64×64` results demonstrate control and color/"
    "shape separation, not photorealism or readable packaging text."
)

st.subheader("Reproducibility and engineering checks")
check_columns = st.columns(2)
with check_columns[0].container(border=True, height="stretch"):
    st.markdown("**Verified contracts**")
    st.markdown(
        "- DDPM coefficients and batched `q_sample`\n"
        "- U-Net forward shapes and finite gradients\n"
        "- PAD/causal masks and attention probabilities\n"
        "- No future-token leakage\n"
        "- Deterministic fixed-seed sampling\n"
        "- Greedy and beam termination\n"
        "- Streamlit render and inference adapters"
    )
with check_columns[1].container(border=True, height="stretch"):
    st.markdown("**Reproduce locally**")
    st.code(
        "conda activate multimodal-caption-diffusion\n"
        "pytest -q\n"
        "streamlit run demo/streamlit_app.py",
        language="bash",
    )
    st.caption(
        "Checkpoints restore their exact config and vocabulary. Missing demo "
        "weights download from the versioned GitHub release with SHA-256 checks."
    )

with st.expander("Limitations that remain", icon=":material/warning:"):
    st.markdown(
        "- The domain has only three correlated product classes.\n"
        "- Text sensitivity does not prove compositional disentanglement.\n"
        "- Caption BLEU is single-reference and the vocabulary is controlled.\n"
        "- The strict tiny-overfit gate for captioning was explicitly skipped.\n"
        "- The 1,000-step ancestral sampler is slow on CPU.\n"
        "- Advanced diffusion checkpoints A1–A11 remain deferred."
    )
