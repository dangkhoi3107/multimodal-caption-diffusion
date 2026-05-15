# Image Captioning and Text-to-Image Diffusion from Scratch

This repository contains a small-scale multimodal generative AI project with two main tasks:

1. **Image Captioning**: generate natural-language descriptions from images.
2. **Text-to-Image Diffusion**: train a text-conditioned diffusion model from scratch to generate images from text prompts.

The goal of this project is to understand and implement the core components behind vision-language and generative AI systems, including visual encoders, text embeddings, U-Net denoising, noise scheduling, and training/evaluation pipelines.

> Project status: under development. The repository is prepared as a portfolio project and will be updated with training code, experiments, and qualitative results.

---

## Project Motivation

Modern multimodal AI systems can connect visual and textual information in both directions:

- **Image → Text**: image captioning helps a model describe visual content using natural language.
- **Text → Image**: text-to-image diffusion generates images conditioned on textual prompts.

This project explores both directions in one repository to build a stronger understanding of multimodal learning and diffusion-based generation.

---

## Main Tasks

### 1. Image Captioning

The image captioning module focuses on generating a caption from an input image.

**Planned pipeline:**

```text
Image
  ↓
Visual Encoder
  ↓
Image Feature Vector
  ↓
Sequence Decoder / Transformer Decoder
  ↓
Generated Caption
```

**Core components:**

- Image preprocessing and augmentation
- Visual encoder for image feature extraction
- Text tokenizer and vocabulary preparation
- Caption decoder for sequence generation
- Training loop with teacher forcing
- Caption evaluation and qualitative examples

---

### 2. Text-to-Image Diffusion from Scratch

The text-to-image module focuses on training a small diffusion model from scratch.

**Planned pipeline:**

```text
Text Prompt
  ↓
Text Encoder / Text Embedding
  ↓
Noise Sampling
  ↓
U-Net Denoising Model
  ↓
Reverse Diffusion Process
  ↓
Generated Image
```

**Core components:**

- Forward diffusion process
- Noise scheduler
- Time-step embedding
- Text conditioning
- U-Net denoising network
- Reverse denoising sampling
- Qualitative image generation results

---

## Repository Structure

```text
multimodal-caption-diffusion/
│
├── configs/                  # Training and experiment configs
├── data/                     # Dataset notes or sample data instructions
├── notebooks/                # Colab / Kaggle experiment notebooks
├── src/
│   ├── captioning/           # Image captioning model and training code
│   ├── diffusion/            # Diffusion model, U-Net, scheduler, sampler
│   └── utils/                # Shared preprocessing and utility functions
│
├── outputs/
│   ├── captions/             # Captioning outputs
│   └── generated_images/     # Text-to-image outputs
│
├── assets/                   # README images, diagrams, and visual results
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Planned Features

- [ ] Build image captioning dataset loader
- [ ] Implement visual encoder
- [ ] Implement caption decoder
- [ ] Train image captioning model
- [ ] Implement diffusion forward process
- [ ] Implement noise scheduler
- [ ] Implement text-conditioned U-Net
- [ ] Train small text-to-image diffusion model from scratch
- [ ] Add qualitative results for captions and generated images
- [ ] Add diagrams and experiment notes

---

## Technologies

- Python
- PyTorch
- torchvision
- NumPy
- OpenCV
- Matplotlib
- Hugging Face Tokenizers / Transformers *(optional for text encoding experiments)*

---

## Expected Outputs

### Image Captioning

Example format:

```text
Input image: product_001.jpg
Generated caption: "A detergent pouch displayed on a shelf."
```

### Text-to-Image Diffusion

Example format:

```text
Prompt: "a small red object on a white background"
Generated image: outputs/generated_images/sample_001.png
```

---

## Portfolio Summary

This project demonstrates hands-on understanding of:

- Vision-language learning
- Image captioning
- Diffusion model training
- Text conditioning
- U-Net denoising
- Noise scheduling
- Image-text dataset preparation
- Qualitative evaluation for generative models

---

## Author

**Pham Nguyen Dang Khoi**  
Computer Science Student, Ho Chi Minh City International University

GitHub: [github.com/dangkhoi3107](https://github.com/dangkhoi3107)
