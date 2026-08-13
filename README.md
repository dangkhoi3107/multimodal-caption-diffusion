# Multimodal Generation From Scratch

An educational PyTorch project that implements two multimodal generation directions from random initialization:

1. **Class/Text → Image** with a pixel-space Denoising Diffusion Probabilistic Model (DDPM).
2. **Image → Text** with a custom CNN encoder and Transformer decoder.

The project prioritizes understanding, correctness, reproducibility, and small verifiable experiments. It does not target Stable Diffusion-level image quality or production deployment.

## Project roadmap

```text
Foundation/Data
  Gaussian toy + COCO polygon → clean 64×64 product crops

Core Diffusion
  forward DDPM → U-Net noise prediction → reverse DDPM
  → class conditioning + CFG → text conditioning

Advanced Diffusion
  Improved DDPM/EDM → DDIM/fast solvers
  → NCSN/Score SDE → autoencoder/LDM → DiT
  → Consistency Models → Flow Matching/Rectified Flow

Captioning
  image → CNN visual tokens → causal Transformer decoder → caption

Research
  evaluation → ablation → failure analysis → reproducible report
```

- [task.md](task.md): source-of-truth checklist, phase gates and evidence requirements.
- [Diffusion curriculum](docs/roadmap/diffusion-curriculum.md): recommended learning order and dependencies.
- [Model catalog](docs/references/model-catalog.md): original papers, official code and online model cards.
- [Repository architecture](docs/architecture/repository-layout.md): file ownership and target structure.
- [AGENTS.md](AGENTS.md): project-wide development and teaching rules.

## What “from scratch” means

The main implementation may use basic PyTorch layers, tensor operations, autograd, optimizers, and data utilities. The following components are implemented directly in this repository:

- DDPM noise schedule, forward process, reverse step, and sampler.
- Sinusoidal timestep embeddings.
- Residual blocks and U-Net.
- Class conditioning and classifier-free guidance.
- Word-level tokenizer, vocabulary, masks, and text encoder.
- Scaled dot-product attention, multi-head attention, and cross-attention.
- CNN image encoder and Transformer caption decoder.
- Training, checkpointing, decoding, and evaluation loops.
- Advanced checkpoints, when selected: DDIM, score matching/Langevin dynamics, SDE/ODE sampling, autoencoder/VAE, latent diffusion, mini-DiT, consistency distillation and flow matching.

The main models do **not** use pretrained weights, Stable Diffusion, Diffusers model components, CLIP, LoRA, DreamBooth, Textual Inversion, pretrained torchvision models, Hugging Face Transformers, or PyTorch's high-level Transformer/MultiheadAttention implementations.

## Current status

- The repository structure, complete core/advanced curriculum and primary-source catalog are defined.
- One COCO product dataset is available under `data/raw/products/`.
- The current dataset contains 104 shelf images and 144 annotated instances of one Lifebuoy SKU.
- Phase 0 preprocessing has not been implemented yet.
- No model has been trained yet.

The active learning checkpoint is F0: a Gaussian-mixture forward-noising experiment. The first data vertical slice is:

```text
one COCO annotation
→ one polygon mask
→ one masked crop
→ aspect-ratio-preserving 64×64 letterbox image
→ visual QA + tests
```

## Repository structure

```text
multimodal-caption-diffusion/
├── AGENTS.md                 # Project-wide implementation and teaching rules
├── task.md                   # Detailed roadmap and checklists
├── docs/
│   ├── roadmap/              # Learning order and dependencies
│   ├── references/           # Papers, official code, model cards
│   └── architecture/         # Module ownership and target tree
├── configs/                  # Reproducible phase/experiment YAML files
├── data/
│   ├── raw/                  # Immutable source datasets; ignored by Git
│   └── processed/            # Reproducible generated datasets; ignored by Git
├── src/
│   ├── data/                 # Parsing, preprocessing, Dataset/DataLoader logic
│   ├── diffusion/            # Scheduler, embeddings, U-Net, trainer, sampler
│   ├── autoencoding/         # Ownership placeholder; implementation starts at A6
│   ├── dit/                  # Ownership placeholder; implementation starts at A8
│   ├── flows/                # Ownership placeholder; implementation starts at A10
│   ├── text/                 # Vocabulary, tokenizer, attention, text encoder
│   ├── captioning/           # CNN encoder, Transformer decoder, decoding
│   ├── metrics/              # BLEU, diversity, and memorization checks
│   └── utils/                # Config, seed, checkpoint, logging utilities
├── scripts/                  # Thin command-line entrypoints
├── tests/                    # Unit, contract, shape, and smoke tests
├── notebooks/                # Inspection and visualization only
├── checkpoints/              # Runtime checkpoints; ignored by Git
├── outputs/                  # Phase-specific experiment artifacts
└── assets/                   # Small documentation figures
```

## Dataset layout

The current raw product export is stored as:

```text
data/raw/products/
└── lifebuoy_handwash_vitamin_protection_400g/
    ├── coco/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    └── source_export.zip
```

`data/raw/` is immutable. Phase 0 will write only to `data/processed/` and `outputs/phase0_data/`.

## Setup

Create and activate a virtual environment, then install the lightweight development dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Training commands will be added phase by phase only after the corresponding implementation and tests exist. The repository intentionally does not advertise commands for unimplemented code.

Online checkpoints linked in the model catalog are references only. They are not downloaded or used as evidence for the from-scratch track.

## Development principles

For every model component:

1. Define its input/output contract and tensor shapes.
2. Add focused tests.
3. Verify finite outputs, losses, and gradients.
4. Overfit one example.
5. Overfit one mini-batch.
6. Only then train the full dataset.

See [AGENTS.md](AGENTS.md) for the complete rules and [task.md](task.md) for phase-specific completion criteria.

## Expected limitations

- A small DDPM trained from scratch on limited data will produce low-resolution, product-like shapes rather than exact packaging.
- Logos and text on product packaging are not expected to be correct.
- Text compositionality will be limited by the size and diversity of the custom dataset.
- A captioning CNN trained from random initialization on Flickr8k is expected to underperform pretrained encoders; that comparison is not the primary goal.

## Author

Pham Nguyen Dang Khoi — Computer Science Student, Ho Chi Minh City International University

[GitHub](https://github.com/dangkhoi3107)
