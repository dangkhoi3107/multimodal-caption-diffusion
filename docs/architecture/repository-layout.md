# Repository architecture and file ownership

## 1. Nguyên tắc

- `task.md` quản lý trạng thái, không chứa implementation.
- `docs/` giải thích và dẫn nguồn.
- `configs/` chứa experiment values, không chứa thuật toán.
- `src/` sở hữu logic tái sử dụng.
- `scripts/` là CLI mỏng.
- `tests/` chứng minh contract.
- `notebooks/` chỉ inspect/visualize.
- `data/raw/` immutable; `data/processed/` sinh lại được.
- `checkpoints/` chứa weights; `outputs/` chứa evidence và report của run.

## 2. Target tree

README ownership placeholders đã được tạo cho các advanced package. Python implementation/config chỉ được tạo just-in-time khi checkpoint tương ứng bắt đầu; cây dưới đây là ownership map, không phải yêu cầu tạo toàn bộ code ngay.

```text
multimodal-caption-diffusion/
├── AGENTS.md
├── task.md
├── README.md
├── requirements.txt
├── docs/
│   ├── README.md
│   ├── roadmap/
│   │   ├── diffusion-curriculum.md
│   │   └── captioning-curriculum.md
│   ├── references/
│   │   └── model-catalog.md
│   └── architecture/
│       └── repository-layout.md
├── configs/
│   ├── data/
│   ├── diffusion/
│   ├── captioning/
│   └── experiments/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── diffusion/
│   │   ├── scheduler.py
│   │   ├── parameterization.py
│   │   ├── embeddings.py
│   │   ├── blocks.py
│   │   ├── unet.py
│   │   ├── conditioning.py
│   │   ├── trainer.py
│   │   ├── samplers/
│   │   │   ├── ddpm.py
│   │   │   ├── ddim.py
│   │   │   └── ode.py
│   │   ├── score/
│   │   │   ├── matching.py
│   │   │   ├── langevin.py
│   │   │   └── sde.py
│   │   └── consistency/
│   │       ├── objective.py
│   │       └── sampler.py
│   ├── autoencoding/
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── vae.py
│   │   └── losses.py
│   ├── dit/
│   │   ├── patching.py
│   │   ├── blocks.py
│   │   └── model.py
│   ├── flows/
│   │   ├── paths.py
│   │   ├── objective.py
│   │   └── solvers.py
│   ├── text/
│   ├── captioning/
│   ├── metrics/
│   └── utils/
├── scripts/
│   ├── prepare_*.py
│   ├── train_*.py
│   ├── sample_*.py
│   └── evaluate_*.py
├── tests/
├── notebooks/
├── checkpoints/<checkpoint>/<run_id>/
└── outputs/<checkpoint>/<run_id>/
```

## 3. Module contracts

### `src/data/`

Owner của parsing, polygon rasterization, crop/letterbox, dataset và collate. Không biết model architecture.

```text
raw annotation/image
→ validated record
→ processed artifact + trace metadata
→ tensor batch
```

### `src/diffusion/scheduler.py`

Owner của coefficients và `q(x_t|x_0)`. Không gọi U-Net, optimizer hoặc filesystem.

### `src/diffusion/parameterization.py`

Chuyển đổi có test giữa epsilon, `x0`, score và velocity parameterizations. Tách khỏi scheduler để các checkpoint hiện đại dùng lại mà không nhét nhiều nhánh vào một class.

### `src/diffusion/unet.py`

Owner của U-Net noise/score predictor. Input/output tensor contract; không chứa loss, reverse sampler hoặc config parsing.

### `src/diffusion/samplers/`

Mỗi sampler sở hữu update equation riêng nhưng dùng chung model prediction interface. DDIM/ODE solver không được duplicate training loop.

### `src/diffusion/score/`

Owner của denoising score matching, Langevin dynamics và continuous SDE abstractions. Chỉ tạo khi score checkpoint bắt đầu.

### `src/autoencoding/`

Owner của image ↔ latent representation. Reconstruction/KL/perceptual choices nằm ở đây; latent diffusion gọi encoder/decoder qua interface, không sửa nội bộ chúng.

### `src/dit/`

Owner của patchify, positional representation, adaLN/modulation và Transformer backbone. Diffusion schedule và sampler vẫn thuộc `src/diffusion/`.

### `src/flows/`

Owner của probability paths, velocity targets và ODE integration. Không gọi các khái niệm flow là beta schedule nếu chúng không phải DDPM schedule.

### `src/text/`

Owner của vocabulary, tokenizer, masks, attention primitives và text encoder. Captioning và text-conditioned diffusion đều có thể dùng module này.

### `src/captioning/`

Owner của CNN visual encoder, causal decoder, caption training và autoregressive decoding. Không phụ thuộc diffusion sampler.

### `src/utils/`

Chỉ chứa cross-cutting utilities thật sự: config, seed, checkpoint, environment capture và artifact logging. Không chuyển model logic vào `utils`.

## 4. Config hierarchy

Khi bắt đầu implementation, config nên được nhóm theo domain:

```text
configs/data/products_64.yaml
configs/diffusion/ddpm_product64.yaml
configs/diffusion/ddim_product64.yaml
configs/diffusion/class_product64.yaml
configs/diffusion/text_product64.yaml
configs/diffusion/latent_product64.yaml
configs/diffusion/dit_product64.yaml
configs/diffusion/consistency_product64.yaml
configs/diffusion/rectified_flow_product64.yaml
configs/captioning/flickr8k.yaml
configs/experiments/<ablation_name>.yaml
```

Mỗi run phải snapshot resolved config vào output. Config inheritance nếu được thêm phải resolve thành một file đầy đủ trước khi train.

## 5. Run artifact contract

```text
outputs/<checkpoint>/<run_id>/
├── config.resolved.yaml
├── environment.json
├── metrics.jsonl
├── command.txt
├── samples/
├── figures/
├── failure_cases/
└── summary.md

checkpoints/<checkpoint>/<run_id>/
├── latest.pt
└── best.pt
```

`summary.md` phải ghi hypothesis, result, gate pass/fail và limitation. Không dùng tên `best.pt` nếu chưa định nghĩa metric lựa chọn trước khi nhìn test set.
