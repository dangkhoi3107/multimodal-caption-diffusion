# Kế hoạch xây dựng Multimodal Generation From Scratch

Tài liệu này là roadmap, checklist và phase-gate chính thức của repository. Mỗi checkbox chỉ được đánh dấu `[x]` khi có bằng chứng kiểm chứng tương ứng. Việc đọc paper, tạo file hoặc viết code nhưng chưa test không đủ để xem là hoàn thành.

Tài liệu hỗ trợ:

- [Lộ trình Diffusion từ cơ bản đến nâng cao](docs/roadmap/diffusion-curriculum.md).
- [Catalog paper, official code và online model](docs/references/model-catalog.md).
- [Kiến trúc repository và trách nhiệm từng file](docs/architecture/repository-layout.md).
- [Lộ trình Image Captioning](docs/roadmap/captioning-curriculum.md).

## 0. Cách sử dụng tài liệu

### Trạng thái checkbox

- `[ ]`: chưa làm hoặc chưa được kiểm chứng.
- `[x]`: đã làm và có bằng chứng.
- Nếu một mục bị chặn, ghi chú ngay dưới mục đó: lý do, bằng chứng và điều kiện để tiếp tục.

### Quy tắc phase gate

- Core Diffusion đi theo `F0 → Phase 0 → Phase 1 → Phase 2 → Phase 3`.
- Advanced Diffusion bắt đầu sau Phase 1 và đi theo dependency ghi ở từng checkpoint; không bắt buộc chạy mọi nhánh để hoàn thành MVP.
- Captioning là track độc lập, có thể bắt đầu sau khi attention/tokenizer contracts của Phase 3 đã ổn hoặc được tự xây riêng trong Phase 4.
- Chỉ chuyển sang checkpoint phụ thuộc khi toàn bộ **Definition of Done** của dependency đã hoàn thành.
- Ngoại lệ phải do người dùng quyết định và ghi rõ dependency còn thiếu, rủi ro và giới hạn kết luận.

### Bằng chứng hợp lệ

- Unit test hoặc smoke test pass.
- Command reproduce có exit code thành công.
- Tensor shape/range được báo cáo.
- Contact sheet, sample grid, loss curve hoặc denoising trajectory được lưu đúng chỗ.
- Metric được tính trên đúng split.
- Overfit test đạt tiêu chí đã định trước.

## 1. Scope đã chốt

### Mục tiêu

Xây dựng hai hướng multimodal generation hoàn toàn từ random initialization:

1. **Class/Text → Image** bằng pixel-space DDPM tự triển khai.
2. **Image → Text** bằng CNN encoder và Transformer decoder tự triển khai.

### Danh sách checkpoint từ cơ bản đến nâng cao

| Thứ tự khuyên dùng | ID | Loại | Câu hỏi cần trả lời | Output/gate chính |
|---:|---|---|---|---|
| 1 | F0 | Math/toy | Gaussian noise, score và vector field được mô phỏng đúng chưa? | Plot 1D/2D + numerical tests |
| 2 | Phase 0 | Data | COCO polygon có trở thành crop sạch và tái lập không? | Crops + masks + metadata + QA |
| 3 | Phase 1 | DDPM core | Có tự train và sample unconditional pixel DDPM không? | DDPM checkpoint + trajectories |
| 4 | A1 | Training design | Cosine/EMA/EDM ideas cải thiện gì dưới cùng budget? | Controlled comparison |
| 5 | A2 | Sampler | DDIM/solver giảm số bước thế nào? | Speed/quality table |
| 6 | Phase 2 | Conditioning | Class ID + CFG có điều khiển class không? | Same-noise different-class grid |
| 7 | Phase 3 | Text conditioning | Prompt/token có điều khiển class và thuộc tính không? | Same-noise prompt swaps |
| 8 | A3 | Spatial conditioning | Mask/edge từ polygon có điều khiển bố cục không? | Control alignment grid |
| 9 | A4 | Score model | NCSN có học `∇x log p_sigma(x)` trên toy data không? | Langevin samples |
| 10 | A5 | SDE/ODE | Reverse SDE/probability-flow ODE có nối được với DDPM không? | Continuous-time trajectories |
| 11 | A6 | Representation | Autoencoder đủ tốt để làm latent space chưa? | Reconstruction/KL report |
| 12 | A7 | Latent diffusion | Diffusion trong latent có tiết kiệm compute mà còn giữ cấu trúc không? | Pixel-vs-latent comparison |
| 13 | A8 | Backbone | Mini-DiT có thay U-Net dưới cùng objective không? | U-Net-vs-DiT comparison |
| 14 | A9 | One/few step | Consistency distillation có tạo mẫu 1/few-step không? | Teacher/student comparison |
| 15 | A10 | Flow | Flow Matching/Rectified Flow có học velocity và ODE path không? | Toy + image flow trajectories |
| 16 | A11 | Architecture study | Có giải thích được SD3/FLUX bằng các trục đã học không? | Technical architecture note |
| 17 | Phase 4 | Image → Text | CNN + Transformer tự xây có caption được không? | Predictions + BLEU |
| 18 | Phase 5 | Research report | Kết quả có tái lập, ablation và failure analysis không? | Final report + demo |

`A1–A11` là track nâng cao. MVP sản phẩm hoàn thành ở Phase 3; không được tuyên bố A-checkpoint đã hoàn thành chỉ vì model cùng tên tồn tại trên mạng.

### Reading map theo checkpoint

Reading map này chỉ ra tài liệu cần đọc trước mỗi checkpoint. Không cần đọc toàn bộ ngay từ đầu. Với mỗi checkpoint, dùng vòng lặp:

```text
đọc tài liệu nền tảng
→ đọc abstract/introduction và phần method của paper gốc
→ viết lại công thức bằng tensor name/shape của project
→ implement toy/vertical slice
→ test và visual QA
→ cuối cùng mới xem official code để đối chiếu
```

Official code và online model chỉ là nguồn tham khảo. Không copy implementation lõi, không tải pretrained weights và không dùng chúng làm kết quả của scratch track.

#### F0 — Thứ tự đọc bắt buộc

F0 là checkpoint duy nhất dùng tiền tố `F`. Sau F0, roadmap dùng `Phase 0–5` cho core/product track và `A1–A11` cho advanced diffusion.

| Thứ tự | Kiến thức | Tài liệu | Sau khi đọc phải làm được |
|---:|---|---|---|
| 1 | Vector, matrix, norm và covariance matrix | [Deep Learning Book — Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html) | Giải thích được shape sample `[N,2]`, mean `[2]` và covariance `[2,2]` |
| 2 | Probability, expectation, variance, conditional probability và Bayes | [Deep Learning Book — Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html) | Tự tính empirical mean/variance và Monte Carlo estimate |
| 3 | Covariance và correlation | [MIT OCW — Covariance and Correlation](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/resources/mitres_6_012s18_l12/) | Giải thích covariance dương, âm và bằng zero |
| 4 | Multivariate Gaussian và reparameterization | [MIT — Multivariate Gaussian Random Variables](https://ocw.mit.edu/courses/6-438-algorithms-for-inference-fall-2014/4f312f9c99b48b35dae961d24f10c471_MIT6_438F14_Lec6.pdf) | Giải thích và mô phỏng `x = mu + L @ epsilon`, `epsilon ~ N(0,I)` |
| 5 | Markov chain | [MIT OCW — Markov Chains I](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-iii/lecture-16/) | Giải thích vì sao forward DDPM dùng `q(x_t | x_{t-1})` |
| 6 | Log-likelihood, entropy, cross-entropy và KL | [Deep Learning Book — Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html) | Phân biệt likelihood, cross-entropy và hai chiều của KL divergence |
| 7 | Chain rule và autograd | [PyTorch — Autograd Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | Theo dõi gradient từ scalar loss về input/parameter |
| 8 | ODE và Euler method | [MIT — Differential Equations Course Notes](https://math.mit.edu/~dunkel/Teach/18.03/2018_CourseNotes.pdf) | Implement `x_next = x + dt * f(t,x)` và kiểm tra error giảm theo step size |
| 9 | Brownian motion và trực giác SDE | [NYU — Brownian Motion Notes](https://math.nyu.edu/~goodman/teaching/StochCalc2004/notes/l5.pdf), [NYU — Ito SDE Notes](https://math.nyu.edu/faculty/goodman/teaching/StochCalc/notes/l9.pdf) | Phân biệt deterministic drift `dt` và stochastic increment `dW`; chưa cần giải tích Itô đầy đủ |
| 10 | Score function | [NCSN paper](https://arxiv.org/abs/1907.05600) | Giải thích `score(x) = grad_x log p(x)` và tính analytic score của Gaussian |
| 11 | Forward diffusion | [DDPM paper](https://arxiv.org/abs/2006.11239) | Giải thích beta schedule, iterative noising và closed-form `q(x_t | x_0)` |

Thứ tự thực hành tương ứng:

```text
probability + Gaussian
→ sinh Gaussian mixture 2D
→ đo empirical mean/covariance

Markov + DDPM
→ tạo beta schedule
→ iterative forward noising
→ closed-form q(x_t | x_0)
→ so sánh thống kê hai cách

score
→ tính analytic score của Gaussian
→ kiểm tra bằng finite difference
→ vẽ vector field

ODE
→ viết Euler integrator
→ giảm step size
→ chứng minh numerical error giảm
```

#### Core/product track

| Checkpoint | Tài liệu nền tảng/paper chính | Official source để đối chiếu | Trọng tâm cần nắm |
|---|---|---|---|
| Phase 0 — Product data | [COCO data format](https://cocodataset.org/#format-data) | [Official COCO API](https://github.com/cocodataset/cocoapi) | `images`, `annotations`, polygon segmentation, bbox, category và trace metadata |
| Phase 1 — Unconditional pixel DDPM | [Diffusion Probabilistic Models — 2015](https://proceedings.mlr.press/v37/sohl-dickstein15.html), [DDPM](https://arxiv.org/abs/2006.11239) | [Official DDPM code](https://github.com/hojonathanho/diffusion) | Schedule, `q_sample`, timestep embedding, epsilon loss, reverse mean/variance và ancestral sampler |
| Phase 2 — Class conditioning + CFG | [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | Không dùng pretrained baseline; đối chiếu phương trình trong paper | Class embedding, condition dropout và kết hợp conditional/unconditional prediction khi sampling |
| Phase 3 — Text conditioning | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [GLIDE](https://arxiv.org/abs/2112.10741) | [Official GLIDE code](https://github.com/openai/glide-text2im) | Vocabulary, attention, text encoder, cross-attention và text CFG; không dùng pretrained CLIP/T5 |

#### Advanced diffusion track

| Checkpoint | Paper chính | Official source để đối chiếu | Trọng tâm cần nắm |
|---|---|---|---|
| A1 — Improved DDPM/EDM | [Improved DDPM](https://proceedings.mlr.press/v139/nichol21a.html), [EDM](https://arxiv.org/abs/2206.00364) | [Improved Diffusion](https://github.com/openai/improved-diffusion), [EDM](https://github.com/NVlabs/edm) | Cosine schedule, reverse variance, EMA, preconditioning và noise-level sampling |
| A2 — DDIM/fast solvers | [DDIM](https://arxiv.org/abs/2010.02502), [PNDM](https://arxiv.org/abs/2202.09778), [DPM-Solver](https://arxiv.org/abs/2206.00927) | [DDIM](https://github.com/ermongroup/ddim), [PNDM](https://github.com/luping-liu/PNDM), [DPM-Solver](https://github.com/LuChengTHU/dpm-solver) | Phân biệt training objective với sampler/solver; so sánh quality theo NFE |
| A3 — Spatial conditioning | [ControlNet](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | [Official ControlNet code](https://github.com/lllyasviel/ControlNet) | Mask/edge control, zero-initialized residual path và spatial alignment |
| A4 — NCSN/score matching | [NCSN](https://arxiv.org/abs/1907.05600) | [Official NCSN code](https://github.com/ermongroup/ncsn) | Denoising score matching, multiple noise levels và annealed Langevin dynamics |
| A5 — Score SDE/ODE | [Score SDE](https://openreview.net/forum?id=PxTIG12RRHS) | [Official JAX code](https://github.com/yang-song/score_sde), [official PyTorch code](https://github.com/yang-song/score_sde_pytorch) | Forward SDE, reverse-time SDE, predictor-corrector và probability-flow ODE |
| A6 — Autoencoder/VAE | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | Paper là nguồn đối chiếu chính; implementation vẫn tự xây | Reconstruction, posterior mean/log-variance, reparameterization và KL gate |
| A7 — Latent diffusion | [Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) | [Official LDM code](https://github.com/CompVis/latent-diffusion) | Tách reconstruction error khỏi diffusion error; normalize latent trước khi train diffusion |
| A8 — Mini-DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | [Official DiT code](https://github.com/facebookresearch/DiT) | Patchify, positional embedding, timestep/condition modulation và unpatchify |
| A9 — Consistency models | [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) | [Official code](https://github.com/openai/consistency_models) | Boundary condition, teacher/student pair và one/few-step generation |
| A10 — Flow Matching/Rectified Flow | [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003) | [Flow Matching](https://github.com/facebookresearch/flow_matching), [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) | Probability path, conditional velocity, vector-field regression, Euler/Heun và reflow |
| A11 — SD3/FLUX study | [SD3 technical paper](https://arxiv.org/abs/2403.03206) | [SD3 model card](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX official code](https://github.com/black-forest-labs/flux) | Map formulation, latent representation, Transformer backbone, text conditioning và solver; không train scale gốc |

#### Image captioning và evaluation

| Checkpoint | Tài liệu | Trọng tâm cần nắm |
|---|---|---|
| Phase 4 — CNN + Transformer captioning | [Show and Tell](https://arxiv.org/abs/1411.4555), [Show, Attend and Tell](https://proceedings.mlr.press/v37/xuc15.html), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Spatial visual tokens, causal self-attention, cross-attention, teacher forcing và autoregressive decoding |
| Phase 4 — BLEU evaluation | [BLEU paper](https://aclanthology.org/P02-1040/) | Modified n-gram precision, brevity penalty và nhiều reference captions |
| Phase 5 — Reproducibility | [PyTorch Reproducibility Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Seed/config/environment capture, deterministic limitations và fair ablation protocol |

Textual Inversion, DreamBooth và LoRA không nằm trong reading gate của F0–Phase 5. Chỉ đọc chúng như phần mở rộng sau khi scratch text-conditioned diffusion đã hoạt động; xem mục fine-tuning trong [model catalog](docs/references/model-catalog.md).

### Ngoài scope

- Ảnh photorealistic 512×512 hoặc Stable Diffusion scale.
- Pretrained weights, CLIP, LoRA, DreamBooth, Textual Inversion.
- General-purpose VLM hoặc visual question answering.
- Đảm bảo logo/chữ trên bao bì chính xác.
- Production deployment, inventory tracking hoặc OCR giá bán.
- Train Stable Diffusion, SD3, Imagen, GLIDE hoặc FLUX ở quy mô gốc.
- Dùng online checkpoint trong [model catalog](docs/references/model-catalog.md) làm kết quả của scratch track.

### Master progress

- [x] F0 — Math and toy distributions.
- [ ] Phase 0 — Product data preparation.
- [ ] Phase 1 — Unconditional pixel DDPM.
- [ ] A1 — Improved DDPM/EDM design experiments.
- [ ] A2 — DDIM and optional fast solvers.
- [ ] Phase 2 — Class-conditioned DDPM + CFG.
- [ ] Phase 3 — Text-conditioned DDPM.
- [ ] A3 — Mask/edge spatial control.
- [ ] A4 — NCSN/score matching.
- [ ] A5 — Score SDE/probability-flow ODE.
- [ ] A6 — Autoencoder/VAE.
- [ ] A7 — Latent diffusion.
- [ ] A8 — Mini-DiT.
- [ ] A9 — Consistency models.
- [ ] A10 — Flow Matching/Rectified Flow.
- [ ] A11 — SD3/FLUX architecture study.
- [ ] Phase 4 — Image captioning.
- [ ] Phase 5 — Evaluation, ablation and report.

## 2. Trạng thái repository hiện tại

- [x] Repository đã có skeleton `src/captioning`, `src/diffusion`, `src/utils`.
- [x] Documentation map và ownership skeleton cho sampler/score/consistency/autoencoding/DiT/flow đã được tạo.
- [x] Dataset sản phẩm COCO đã được tải và giữ trong `data/raw/products/`.
- [x] Dataset hiện tại có `train`, `valid`, `test` không trùng source image.
- [x] Dataset hiện tại có 104 ảnh và 144 instance của một SKU Lifebuoy.
- [x] Polygon COCO hiện tại có segmentation, bbox, area và category hợp lệ.
- [ ] Chưa có processed crop dataset.
- [ ] Chưa có model diffusion.
- [ ] Chưa có model captioning.
- [ ] Chưa có training/evaluation pipeline.

## 3. Cấu trúc repository mục tiêu

```text
multimodal-caption-diffusion/
├── AGENTS.md                         # Quy tắc làm việc chung cho Codex/dev
├── task.md                           # Roadmap, checklist và phase gates
├── README.md                         # Tổng quan project và cách bắt đầu
├── requirements.txt
├── docs/
│   ├── roadmap/                      # Thứ tự học và dependency
│   ├── references/                   # Paper/code/model links
│   └── architecture/                 # Module ownership và target tree
├── configs/
│   ├── phase0_data.yaml
│   ├── phase1_unconditional.yaml
│   ├── phase2_class_conditional.yaml
│   ├── phase3_text_conditional.yaml
│   └── phase4_captioning.yaml
├── data/
│   ├── raw/                          # Immutable, không commit
│   │   ├── products/
│   │   └── captioning/
│   └── processed/                    # Sinh lại được, không commit
│       ├── products_64/
│       └── captioning/
├── src/
│   ├── data/                         # Parse, preprocess và Dataset
│   ├── diffusion/                    # Scheduler, U-Net, trainer, sampler
│   ├── autoencoding/                 # Encoder/decoder/VAE cho latent track
│   ├── dit/                          # Transformer diffusion backbone
│   ├── flows/                        # Flow paths, velocity loss, ODE solver
│   ├── text/                         # Vocabulary, tokenizer, text encoder
│   ├── captioning/                   # CNN encoder, decoder, decoding
│   ├── metrics/                      # BLEU, diversity, memorization
│   └── utils/                        # Seed, config, checkpoint, logging
├── scripts/                          # CLI entrypoints mỏng
├── tests/                            # Unit, contract và smoke tests
├── notebooks/                        # Chỉ inspect/visualize, không chứa core logic
├── checkpoints/                      # Không commit
├── outputs/                          # Artifact theo phase
└── assets/                           # Hình/tài liệu nhỏ dùng trong README
```

Không tạo toàn bộ file implementation ngay từ đầu. File của phase nào chỉ được tạo khi bắt đầu vertical slice tương ứng.

Chi tiết ownership và tên file nâng cao nằm ở [repository layout](docs/architecture/repository-layout.md).

---

# Checkpoint F0 — Math and Toy Distributions

## F0.1 Mục tiêu

Chứng minh các primitive toán học bằng scalar/2D data trước khi gắn vào ảnh và U-Net.

## F0.2 Checklist kiến thức

- [ ] Mean, variance, standard deviation và covariance.
- [ ] Gaussian 1D/multivariate và sampling bằng reparameterization.
- [ ] Expectation và Monte Carlo estimate.
- [ ] Conditional probability, Bayes rule và Markov chain.
- [ ] Log-likelihood, cross-entropy và KL divergence.
- [ ] Gradient của log-density; phân biệt `∇p(x)` và `∇log p(x)`.
- [ ] Đạo hàm/chain rule đủ để đọc epsilon loss.
- [ ] ODE intuition: state thay đổi theo deterministic vector field.
- [ ] SDE intuition: ODE cộng stochastic increment; chưa cần giải tích Itô đầy đủ.

## F0.3 Checklist implementation

- [x] Tạo Gaussian mixture 2D bằng NumPy/PyTorch với seed cố định.
- [x] Kiểm tra empirical mean/covariance gần giá trị cấu hình.
- [x] Viết linear beta schedule trên vector `[T]`.
- [x] Mô phỏng iterative forward noising.
- [x] Viết closed-form sample `x_t` và so sánh phân phối với iterative version.
- [x] Vẽ scatter tại ít nhất 5 timestep.
- [x] Tính analytic score của Gaussian 2D và kiểm tra bằng finite difference.
- [x] Vẽ score vector field trên grid.
- [x] Viết Euler integrator cho một ODE đơn giản và test convergence khi giảm step size.
- [x] Lưu công thức ↔ tensor mapping trong experiment summary.

## F0.4 Output bắt buộc

```text
outputs/f0_math_toy/<run_id>/
├── config.resolved.yaml
├── gaussian_statistics.json
├── forward_noising_2d.png
├── score_field_2d.png
├── ode_trajectory.png
└── summary.md
```

## F0.5 Definition of Done

- [x] Numerical statistics tests pass với tolerance được ghi rõ.
- [x] Closed-form và iterative forward process có distribution statistics phù hợp.
- [x] Score finite-difference test pass ngoài điểm density quá nhỏ.
- [x] Euler test cho thấy error giảm khi step size giảm.
- [x] Có thể giải thích từng ký hiệu bằng tên tensor và shape.

---

# Phase 0 — Product Data Preparation

## 0.1 Mục tiêu

Chuyển dữ liệu COCO instance segmentation:

```text
shelf image + polygon annotation
```

thành dataset ảnh sản phẩm độc lập:

```text
RGB product crop [3, 64, 64] + class_id + trace metadata
```

Phase này chưa train model. Output phải deterministic khi dùng cùng config và seed.

## 0.2 Chuẩn bị trước khi code

- [ ] Xác nhận đường dẫn raw dataset trong config, không hardcode.
- [ ] Ghi lại số ảnh, số annotation và class theo từng split.
- [ ] Xác nhận `category_id=0` nếu chỉ là supercategory sẽ không trở thành class train.
- [ ] Chốt output size ban đầu: `64×64`.
- [ ] Chốt margin quanh bbox: bắt đầu `5%` hoặc `10%`.
- [ ] Chốt background baseline: trắng hoàn toàn trước, random neutral là ablation sau.
- [ ] Chốt tiêu chí bỏ crop: polygon rỗng, bbox không hợp lệ, crop quá nhỏ hoặc ảnh đọc lỗi.

## 0.3 File và idea dự kiến

### `configs/phase0_data.yaml`

**Idea:** mô tả toàn bộ preprocessing để dataset processed có thể sinh lại. Không chứa logic.

Các field tối thiểu:

```yaml
raw_root: data/raw/products/lifebuoy_handwash_vitamin_protection_400g/coco
output_root: data/processed/products_64
image_size: 64
margin_ratio: 0.10
background_mode: white
seed: 42
```

### `src/data/coco.py`

**Idea:** chỉ chịu trách nhiệm đọc và index COCO; không crop hoặc resize ảnh.

Function dự kiến:

- `load_coco(path: Path) -> CocoDocument`
  - Input: đường dẫn `_annotations.coco.json`.
  - Output: cấu trúc typed chứa images, annotations, categories.
  - Validate: ID duy nhất, file name tồn tại trong JSON, segmentation có số tọa độ chẵn.
- `build_coco_indexes(document) -> CocoIndexes`
  - Tạo map `image_id -> image`, `image_id -> annotations`, `category_id -> category`.
  - Không đọc pixel ảnh.
- `iter_instances(indexes) -> Iterator[ProductInstance]`
  - Yield từng annotation cùng thông tin source image/category.

### `src/data/product_preprocessing.py`

**Idea:** chứa các pure transform cho một instance; không biết về CLI hoặc toàn bộ dataset.

Function dự kiến:

- `polygon_to_mask(polygons, height, width) -> np.ndarray`
  - Output shape `[H, W]`, dtype `uint8`, giá trị `{0, 1}`.
- `mask_bounds(mask) -> tuple[int, int, int, int]`
  - Output convention phải ghi rõ: `(left, top, right_exclusive, bottom_exclusive)`.
- `expand_bounds(bounds, margin_ratio, image_size) -> bounds`
  - Clip vào biên ảnh, không tạo tọa độ âm.
- `apply_mask(image, mask, background) -> np.ndarray`
  - Giữ pixel object; thay ngoài mask bằng background.
- `letterbox_square(image, size, fill) -> np.ndarray`
  - Giữ aspect ratio, resize cạnh dài về `size`, padding đối xứng.
- `prepare_instance(image, annotation, config) -> PreparedCrop`
  - Orchestrate các pure transform cho một instance.

### `src/data/product_dataset.py`

**Idea:** load processed PNG + metadata cho DataLoader; không parse COCO raw.

Class dự kiến:

- `ProductImageDataset`
  - Input metadata JSONL.
  - Output Phase 1: `image` tensor `[3, 64, 64]`, range `[-1, 1]`.
  - Output Phase 2+: thêm `class_id`, `class_name`.
  - Validate RGB, shape và finite pixel.

### `scripts/prepare_products.py`

**Idea:** CLI mỏng gọi parser + preprocessing, ghi PNG/JSONL và report. Không chứa thuật toán mask/resize.

Luồng:

1. Parse `--config`.
2. Load config và seed.
3. Lặp từng split.
4. Gọi `prepare_instance`.
5. Ghi ảnh và metadata atomically.
6. Ghi summary/error report.

### `scripts/inspect_products.py`

**Idea:** tạo contact sheet và thống kê để người dùng kiểm tra trực quan trước training.

### Tests dự kiến

- `tests/test_coco.py`: index ID, category mapping, malformed segmentation.
- `tests/test_product_preprocessing.py`: mask shape, bounds, clipping, letterbox không méo.
- `tests/test_product_dataset.py`: tensor shape/range/label.

## 0.4 Checklist triển khai

### A. COCO parser

- [ ] Tạo `phase0_data.yaml` với path tương đối.
- [ ] Tạo dataclass/type cho image, annotation và category.
- [ ] Implement `load_coco`.
- [ ] Implement `build_coco_indexes`.
- [ ] Báo lỗi rõ khi ảnh hoặc annotation thiếu field.
- [ ] Test category mapping và instance count trên fixture nhỏ.

### B. Polygon và crop

- [ ] Implement polygon → binary mask.
- [ ] Test polygon đơn giản bằng hình chữ nhật biết trước diện tích.
- [ ] Hỗ trợ một annotation có nhiều polygon.
- [ ] Implement bounds convention và test off-by-one.
- [ ] Implement margin và clip vào image bounds.
- [ ] Implement masked composite trên nền trắng.
- [ ] Implement resize giữ aspect ratio.
- [ ] Implement padding vuông đối xứng.
- [ ] Xác nhận output `[64, 64, 3]`, RGB, `uint8`.

### C. Dataset generation

- [ ] Tạo tên file output deterministic theo split/image/annotation ID.
- [ ] Ghi `train.jsonl`, `valid.jsonl`, `test.jsonl`.
- [ ] Metadata có `source_image`, `image_id`, `annotation_id`, `class_id`, `class_name`.
- [ ] Metadata có thông tin bounds và preprocessing config/version.
- [ ] Giữ nguyên split gốc.
- [ ] Ghi summary: số processed, skipped và lỗi theo split/class.
- [ ] Không ghi bất kỳ output nào vào `data/raw/`.

### D. QA

- [ ] Sinh contact sheet tối thiểu 32 crop train.
- [ ] Sinh contact sheet cho toàn bộ valid/test nếu số lượng nhỏ.
- [ ] Kiểm tra thủ công object không bị cắt nắp/cạnh quan trọng.
- [ ] Kiểm tra không kéo méo aspect ratio.
- [ ] Kiểm tra mask không giữ vùng background lớn bất thường.
- [ ] Tìm duplicate/gần duplicate và ghi nhận, chưa tự ý xóa.
- [ ] Load một DataLoader batch.
- [ ] Báo cáo batch shape `[B, 3, 64, 64]`, dtype và min/max.
- [ ] Test deterministic: chạy hai lần cùng seed/config cho cùng checksum metadata.

## 0.5 Output bắt buộc

```text
data/processed/products_64/
├── train/
├── valid/
├── test/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── classes.json
└── preprocessing_summary.json

outputs/phase0_data/
├── contact_sheet_train.png
├── contact_sheet_valid.png
├── contact_sheet_test.png
└── qa_report.json
```

## 0.6 Definition of Done

- [ ] Toàn bộ unit test Phase 0 pass.
- [ ] Không có output crop sai shape/dtype/range.
- [ ] DataLoader trả `[B, 3, 64, 64]` trong `[-1, 1]`.
- [ ] Contact sheet đã được người dùng hoặc dev kiểm tra.
- [ ] Tất cả crop trace ngược được về raw annotation.
- [ ] Dataset processed sinh lại được từ một command và config.

---

# Phase 1 — Unconditional Pixel-space DDPM

## 1.1 Mục tiêu

Xây DDPM không có class/text condition:

```text
Gaussian noise [B, 3, 64, 64] → product-like image [B, 3, 64, 64]
```

Model học dự đoán noise `epsilon` từ `x_t` và timestep `t`.

## 1.2 Kiến thức/công thức phải hiểu trước

- [ ] Giải thích `beta_t`, `alpha_t = 1 - beta_t`.
- [ ] Giải thích `alpha_bar_t = product(alpha_1 ... alpha_t)`.
- [ ] Mapping công thức sang tensor shape `[T]`.
- [ ] Giải thích sampling trực tiếp:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

- [ ] Giải thích target `epsilon` và MSE loss.
- [ ] Giải thích vì sao U-Net cần timestep embedding.
- [ ] Giải thích reverse mean, variance và noise tại `t > 0`.

## 1.3 File và idea dự kiến

### `configs/phase1_unconditional.yaml`

Chứa image size, channels, channel multipliers, timestep count, beta range, batch size, LR, epochs/steps, seed, checkpoint/sample interval.

### `src/diffusion/scheduler.py`

**Idea:** giữ toàn bộ toán DDPM; không biết model/training loop.

Function/class dự kiến:

- `linear_beta_schedule(num_steps, beta_start, beta_end) -> Tensor[T]`.
- `extract(values, timesteps, target_shape) -> Tensor[B, 1, 1, 1]`.
- `DDPMScheduler.q_sample(x_0, t, noise) -> x_t`.
- `DDPMScheduler.p_mean_variance(model_output, x_t, t) -> stats`.
- `DDPMScheduler.p_sample(model, x_t, t) -> x_prev`.

Invariant: coefficients cùng device/dtype với input; `t` shape `[B]`, dtype integer.

### `src/diffusion/embeddings.py`

**Idea:** chuyển timestep integer `[B]` thành vector liên tục `[B, D]`.

- `sinusoidal_timestep_embedding(t, dim, max_period) -> [B, D]`.
- `TimestepMLP`: `[B, D] -> [B, time_dim]`.

### `src/diffusion/blocks.py`

**Idea:** primitive tái sử dụng của U-Net.

- `ResidualBlock.forward(x, time_emb) -> y`.
- `Downsample.forward([B,C,H,W]) -> [B,C,H/2,W/2]`.
- `Upsample.forward([B,C,H,W]) -> [B,C,H*2,W*2]`.
- Optional `SpatialSelfAttention` chỉ thêm sau baseline chạy.

### `src/diffusion/unet.py`

**Idea:** lắp blocks thành symmetric U-Net, quản lý skip connections và channel contracts.

- `UNet.forward(x_t, timesteps) -> predicted_noise`.
- Input/output cùng shape `[B, 3, 64, 64]`.
- Không chứa scheduler hoặc loss.

### `src/diffusion/trainer.py`

**Idea:** một training/validation step có thể test độc lập; không parse CLI.

- `compute_diffusion_loss(model, scheduler, x_0, t, noise) -> scalar`.
- `train_epoch(...) -> metrics`.
- `validate_epoch(...) -> metrics`.

### `src/diffusion/sampler.py`

**Idea:** reverse loop và lưu intermediate states; không load config từ CLI.

- `sample_ddpm(model, scheduler, shape, generator, return_trajectory) -> sample`.

### CLI

- `scripts/train_phase1.py`: config → data/model/scheduler/trainer.
- `scripts/sample_phase1.py`: checkpoint + seed → grid/trajectory.

### Tests

- `tests/test_scheduler.py`.
- `tests/test_embeddings.py`.
- `tests/test_diffusion_blocks.py`.
- `tests/test_unet.py`.
- `tests/test_sampler.py`.

## 1.4 Checklist triển khai

### A. Scheduler

- [ ] Implement linear beta schedule.
- [ ] Assert `0 < beta_t < 1` và monotonic cho config baseline.
- [ ] Precompute alphas, cumulative products và posterior coefficients.
- [ ] Implement batch-safe `extract`.
- [ ] Implement `q_sample` với noise truyền vào để test deterministic.
- [ ] Test `q_sample` shape/device/dtype.
- [ ] Visualize cùng một ảnh tại `t = 0, 100, 300, 500, 999`.
- [ ] Xác nhận timestep cuối gần Gaussian noise.

### B. Timestep embedding

- [ ] Implement sin/cos embedding không dùng thư viện model cấp cao.
- [ ] Xử lý embedding dimension lẻ.
- [ ] Test cùng timestep cho cùng vector.
- [ ] Test timestep khác tạo vector khác.
- [ ] Implement two-layer MLP.

### C. U-Net blocks

- [ ] Implement normalization + activation + conv order đã chọn và document.
- [ ] Project time embedding vào channel dimension.
- [ ] Implement residual shortcut khi input/output channels khác nhau.
- [ ] Test gradient đi qua time projection.
- [ ] Implement downsample.
- [ ] Implement upsample.
- [ ] Test từng block với batch > 1.

### D. Full U-Net

- [ ] Vẽ/báo cáo channel và resolution ở từng stage.
- [ ] Implement down path và lưu skip tensors.
- [ ] Implement middle blocks.
- [ ] Implement up path và concatenate/add skip đúng channel.
- [ ] Output conv trả đúng 3 channels.
- [ ] Test input/output `[2, 3, 64, 64]`.
- [ ] Test loss backward tạo finite gradients.
- [ ] Báo cáo parameter count.
- [ ] Chưa thêm attention nếu baseline chưa pass.

### E. Training correctness ladder

- [ ] Implement random timestep per sample.
- [ ] Implement random Gaussian noise cùng shape `x_0`.
- [ ] Implement epsilon-prediction MSE.
- [ ] Overfit **một ảnh**; đặt tiêu chí loss/sample trước khi chạy.
- [ ] Lưu fixed-noise sample trong one-image experiment.
- [ ] Overfit **một mini-batch 8–16 ảnh**.
- [ ] Xác nhận không NaN/Inf ở loss và gradient.
- [ ] Chỉ sau đó train toàn train split.
- [ ] Validation chỉ đo trên valid split, không update parameter.

### F. Reverse sampling

- [ ] Implement reverse mean.
- [ ] Implement posterior variance.
- [ ] Không thêm random noise khi `t=0`.
- [ ] Implement full loop `T-1 → 0`.
- [ ] Hỗ trợ fixed `torch.Generator`.
- [ ] Test cùng seed/checkpoint tạo cùng output.
- [ ] Lưu denoising trajectory có số frame giới hạn.
- [ ] Clamp/unnormalize chỉ ở visualization boundary, không giấu lỗi trong model.

### G. Experiment hygiene

- [ ] Config lưu cùng run.
- [ ] Checkpoint có model, optimizer, epoch/step, config, seed.
- [ ] Resume checkpoint được smoke test.
- [ ] Loss curve được lưu.
- [ ] Sample grid dùng fixed seed qua các epoch.
- [ ] Ghi hardware, duration và git commit nếu có.

## 1.5 Output bắt buộc

```text
checkpoints/phase1_unconditional/<run_id>/
├── latest.pt
└── best.pt

outputs/phase1_unconditional/<run_id>/
├── config.yaml
├── metrics.jsonl
├── loss_curve.png
├── fixed_seed_samples/
├── final_samples.png
└── denoising_trajectory.png
```

## 1.6 Definition of Done

- [ ] Scheduler/unit/shape/gradient tests pass.
- [ ] Forward diffusion visualization đúng trực quan.
- [ ] Model overfit được một ảnh.
- [ ] Model overfit được mini-batch.
- [ ] Reverse sampler deterministic với fixed seed và không NaN.
- [ ] Full dataset training có checkpoint, curve và samples.
- [ ] Sample có cấu trúc/màu sắc khác pure noise, không yêu cầu logo đúng.

---

# Phase 2 — Class-Conditioned DDPM

## 2.1 Mục tiêu

Mở rộng U-Net:

```text
UNet(x_t, timestep, class_id) -> predicted_noise
```

và chứng minh class condition ảnh hưởng kết quả khi giữ nguyên initial noise.

## 2.2 Điều kiện dữ liệu

- [ ] Phase 1 đạt Definition of Done.
- [ ] Có tối thiểu 2 class; mục tiêu tốt hơn là 3–5 class.
- [ ] Taxonomy cùng cấp độ, ưu tiên SKU-level.
- [ ] Mỗi class có target tối thiểu đã ghi rõ trước thu thập.
- [ ] Mọi object thuộc class mục tiêu xuất hiện trong ảnh đã được annotate nhất quán.
- [ ] Thống kê train/valid/test và imbalance theo class.
- [ ] Không coi augmentation từ cùng source là sample độc lập khi split.

## 2.3 File và idea dự kiến

### `configs/phase2_class_conditional.yaml`

Kế thừa rõ các hyperparameter Phase 1 và thêm `num_classes`, `condition_dropout`, `guidance_scales`, class mapping.

### `src/diffusion/conditioning.py`

**Idea:** biến class/text condition thành vector có cùng dimension với time embedding.

- `ClassConditioner(num_classes, embedding_dim)`.
- Reserved `null_class_id` cho classifier-free guidance.
- `drop_condition(class_ids, probability, generator)`.
- Không chứa sampling formula.

### Thay đổi `src/diffusion/unet.py`

- `ConditionalUNet.forward(x_t, timesteps, class_ids)`.
- Combine baseline: `time_embedding + class_embedding`.
- Giữ unconditional U-Net có thể chạy hoặc dùng null condition rõ ràng.

### Thay đổi `src/diffusion/sampler.py`

- `guided_noise_prediction(...)` tính:

```text
eps = eps_uncond + scale * (eps_cond - eps_uncond)
```

- Không duplicate toàn bộ reverse loop.

### CLI/tests

- `scripts/train_phase2.py`.
- `scripts/sample_phase2.py`.
- `tests/test_conditioning.py`.
- Mở rộng U-Net/sampler tests.

## 2.4 Checklist triển khai

### A. Data expansion

- [ ] Chốt danh sách class và mapping ID ổn định.
- [ ] Chuẩn hóa class name theo `brand_category_variant_size`.
- [ ] Thu thập/annotate thêm class.
- [ ] Chạy lại Phase 0 cho multi-class dataset.
- [ ] Review contact sheet riêng từng class.
- [ ] Kiểm tra imbalance và quyết định sampler/weight nếu cần.

### B. Class conditioning

- [ ] Implement class embedding.
- [ ] Implement null class embedding.
- [ ] Implement condition dropout ở training.
- [ ] Combine class/time embedding và document shape `[B, D]`.
- [ ] Test class IDs ngoài range báo lỗi.
- [ ] Test null/real condition đều forward được.
- [ ] Test gradient tới class embedding.

### C. Training

- [ ] Khởi tạo random toàn bộ model; không load Phase 1 weights cho experiment chính.
- [ ] Có thể chạy experiment phụ warm-start nhưng phải ghi rõ, không thay thế scratch run.
- [ ] Overfit mini-batch có ít nhất một sample mỗi class.
- [ ] Kiểm tra model không bỏ qua class bằng cùng noise khác class.
- [ ] Train full multi-class dataset.
- [ ] Log loss tổng và phân tích sample theo từng class.

### D. Classifier-free guidance

- [ ] Implement hai forward cond/uncond khi sampling.
- [ ] Test `scale=0` tương đương unconditional prediction.
- [ ] Test `scale=1` tương đương conditional prediction theo công thức chọn.
- [ ] Sinh comparison grid cho scale `0, 1, 3, 5`.
- [ ] Ghi nhận scale quá cao làm giảm diversity/artifact ra sao.

### E. Evaluation

- [ ] Cùng seed, khác class grid.
- [ ] Cùng class, khác seed grid.
- [ ] Kiểm tra class collapse về class phổ biến.
- [ ] Train classifier nhỏ từ scratch nếu cần class-consistency metric.
- [ ] Kiểm tra nearest training crop để phát hiện memorization.
- [ ] Báo cáo limitation khi class có quá ít dữ liệu.

## 2.5 Output bắt buộc

```text
outputs/phase2_class_conditional/<run_id>/
├── per_class_samples/
├── same_seed_different_class.png
├── same_class_different_seed.png
├── guidance_scale_comparison.png
├── metrics.jsonl
└── class_distribution.json
```

## 2.6 Definition of Done

- [ ] Có từ 2 class trở lên với processed dataset hợp lệ.
- [ ] Conditional shape/gradient/unit tests pass.
- [ ] Mini-batch multi-class overfit thành công.
- [ ] Cùng noise, class khác tạo thay đổi có hệ thống.
- [ ] CFG hoạt động đúng các boundary scale đã test.
- [ ] Không luôn sinh class phổ biến nhất.
- [ ] Có memorization check và failure cases.

---

# Phase 3 — Text-Conditioned DDPM

## 3.1 Mục tiêu

Chuyển từ class ID sang prompt ngắn:

```text
"a white lifebuoy hand wash bottle"
    -> text encoder -> text condition -> U-Net -> image
```

Baseline text condition dùng pooled text vector. Cross-attention chỉ là extension sau khi baseline hoạt động.

## 3.2 Điều kiện dữ liệu

- [ ] Phase 2 đạt Definition of Done.
- [ ] Mỗi ảnh có caption mô tả thuộc tính quan sát được.
- [ ] Caption không chỉ khác nhau ở class nếu muốn học compositionality.
- [ ] Có controlled vocabulary nhỏ: class, màu, dạng bao bì, background.
- [ ] Không gán thuộc tính không quan sát được.
- [ ] Vocabulary chỉ build từ train split.

## 3.3 File và idea dự kiến

### `configs/phase3_text_conditional.yaml`

Thêm vocab/min frequency/max length/text dimension/text layers/text dropout và prompt dropout.

### `src/text/vocabulary.py`

**Idea:** ánh xạ token ↔ ID ổn định và serialize được.

- Special IDs: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`.
- `build_vocabulary(train_captions, min_frequency)`.
- `save/load` giữ nguyên ID.

### `src/text/tokenizer.py`

**Idea:** normalization và word tokenization deterministic; không phụ thuộc pretrained tokenizer.

- `normalize_text(text) -> str`.
- `tokenize(text) -> list[str]`.
- `encode(text, vocab, max_length) -> token_ids`.
- `decode(token_ids, vocab, stop_at_eos=True) -> str`.

### `src/text/attention.py`

**Idea:** attention primitives dùng chung; triển khai từ tensor ops/Linear.

- `scaled_dot_product_attention(q, k, v, mask)`.
- `MultiHeadAttention` tự reshape heads và merge output.

### `src/text/encoder.py`

**Idea:** token embedding + positional encoding + encoder blocks.

- Output token states `[B, L, D]`.
- Baseline pooled condition `[B, D]` dùng masked mean.
- Không dùng `nn.Transformer*` hoặc `nn.MultiheadAttention`.

### Diffusion integration

- `TextConditioner` project pooled text vào `time_embedding_dim`.
- `TextConditionalUNet.forward(x_t, t, token_ids, padding_mask)`.
- Prompt dropout/null text cho classifier-free guidance.

### CLI/tests

- `scripts/train_phase3.py`, `scripts/sample_phase3.py`.
- `tests/test_vocabulary.py`, `tests/test_tokenizer.py`.
- `tests/test_attention.py`, `tests/test_text_encoder.py`.

## 3.4 Checklist triển khai

### A. Caption metadata

- [ ] Định nghĩa caption schema.
- [ ] Sinh/viết caption cho từng crop.
- [ ] Review caption-class consistency.
- [ ] Thống kê caption length và token frequency.
- [ ] Chốt max sequence length dựa trên train distribution.

### B. Tokenizer/vocabulary

- [ ] Implement normalization.
- [ ] Implement word splitting/punctuation policy.
- [ ] Build vocab chỉ từ train captions.
- [ ] Thêm special tokens với ID cố định.
- [ ] Implement encode BOS/EOS/pad/truncate.
- [ ] Implement decode và bỏ PAD/BOS/EOS đúng cách.
- [ ] Round-trip tests.
- [ ] Unknown-token test trên valid/test.

### C. Attention/text encoder

- [ ] Implement scaled dot-product attention.
- [ ] Test output shape và attention weights sum gần 1.
- [ ] Implement mask broadcast rõ ràng.
- [ ] Implement multi-head split/merge.
- [ ] Implement sinusoidal hoặc learned positional encoding từ scratch.
- [ ] Implement feed-forward và encoder block.
- [ ] Implement padding-aware mean pooling.
- [ ] Test output không đổi do giá trị ở PAD position.
- [ ] Test finite gradients.

### D. Text conditioning

- [ ] Project pooled text condition về time dimension.
- [ ] Kết hợp time/text embedding trong ResBlocks.
- [ ] Implement null prompt/prompt dropout.
- [ ] Overfit controlled mini-batch với nhiều prompt.
- [ ] Cùng initial noise, đổi class word.
- [ ] Cùng initial noise, đổi color/package word.
- [ ] Kiểm tra model có bỏ qua non-class words không.

### E. Optional cross-attention extension

- [ ] Chỉ bắt đầu sau pooled baseline Definition of Done cơ bản.
- [ ] Visual feature `[B, HW, C]` làm query.
- [ ] Text states `[B, L, D]` làm key/value.
- [ ] Padding mask chặn PAD key.
- [ ] Chỉ thêm tại middle block trước.
- [ ] So sánh pooled vs cross-attention bằng cùng config/seed.
- [ ] Lưu attention visualization nếu diễn giải được.

### F. Evaluation

- [ ] Same seed/different prompt grid.
- [ ] Same prompt/different seed diversity grid.
- [ ] Test prompt hoán đổi một thuộc tính.
- [ ] Test prompt có `<UNK>`.
- [ ] Ghi failure khi từ bị bỏ qua hoặc class bleed.
- [ ] Memorization check.

## 3.5 Output bắt buộc

```text
outputs/phase3_text_conditional/<run_id>/
├── vocabulary.json
├── prompt_samples/
├── same_seed_prompt_comparison.png
├── attribute_swap_comparison.png
├── metrics.jsonl
└── failure_cases.json
```

## 3.6 Definition of Done

- [ ] Tokenizer/vocab/attention/text encoder tests pass.
- [ ] Text-conditioned mini-batch overfit thành công.
- [ ] Class word ảnh hưởng kết quả với same seed.
- [ ] Ít nhất một thuộc tính ngoài class có ảnh hưởng đo/quan sát được, hoặc limitation được chứng minh rõ.
- [ ] Prompt CFG hoạt động và có comparison.
- [ ] Model chính vẫn random initialization, không pretrained text encoder.

---

# Advanced Diffusion Track

Track này dùng ID `A1–A11` để không làm thay đổi output contract của Phase 0–5 đã tồn tại. Chỉ A-checkpoint có dependency đã đạt gate mới được bắt đầu. Paper/code/model links đầy đủ nằm ở [model catalog](docs/references/model-catalog.md).

---

# A1 — Improved DDPM and EDM Design Experiments

## A1.1 Dependency và câu hỏi

- Dependency: Phase 1 đạt Definition of Done.
- Câu hỏi: schedule, variance, parameterization, preconditioning và EMA thay đổi training/sampling như thế nào dưới cùng data và compute budget?
- Nguồn chính: [Improved DDPM](https://proceedings.mlr.press/v139/nichol21a.html), [official code](https://github.com/openai/improved-diffusion), [EDM](https://arxiv.org/abs/2206.00364), [official EDM code](https://github.com/NVlabs/edm).

## A1.2 File/idea dự kiến

- Mở rộng `src/diffusion/scheduler.py`: cosine schedule; không chứa model.
- `src/diffusion/parameterization.py`: chuyển đổi có test giữa epsilon, `x_0`, score và velocity.
- `src/utils/ema.py`: cập nhật shadow weights; không thay optimizer state.
- `configs/experiments/`: mỗi ablation thay đúng một yếu tố.

## A1.3 Checklist

- [ ] Implement cosine schedule và test beta finite, positive, đúng length.
- [ ] Vẽ signal-to-noise ratio của linear và cosine schedule.
- [ ] Implement EMA; test update bằng model nhỏ biết trước.
- [ ] Sample raw weights và EMA weights bằng cùng seed.
- [ ] Tách epsilon/`x_0` conversion thành pure functions.
- [ ] Test round-trip `epsilon → x_0 → epsilon` trong tolerance.
- [ ] Đọc và giải thích learned reverse variance; chưa implement trước cosine/EMA baseline.
- [ ] Optional: implement learned variance + hybrid VLB/MSE objective.
- [ ] Đọc EDM preconditioning; mapping `c_skip`, `c_out`, `c_in`, `c_noise` sang tensor code.
- [ ] Optional: implement mini-EDM experiment riêng, không silently thay Phase 1 DDPM.
- [ ] Chạy linear vs cosine cùng seed, model, data, steps và optimizer.
- [ ] Chạy raw vs EMA sampling cùng checkpoint step.
- [ ] Ghi runtime, peak memory, loss curve và fixed-seed samples.

## A1.4 Output và gate

```text
outputs/a1_improved_design/<run_id>/
├── schedule_snr.png
├── linear_vs_cosine.png
├── raw_vs_ema.png
├── ablation.csv
└── summary.md
```

- [ ] Schedule/parameterization/EMA tests pass.
- [ ] Có ít nhất hai controlled comparisons.
- [ ] Kết luận phân biệt observation với inference; không chỉ chọn ảnh đẹp nhất.
- [ ] Learned variance/EDM optional được ghi rõ là done hoặc deferred.

---

# A2 — DDIM and Optional Fast Solvers

## A2.1 Dependency và câu hỏi

- Dependency: Phase 1 reverse DDPM sampler đúng và deterministic theo seed.
- Câu hỏi: có giảm số network evaluations mà giữ cấu trúc mẫu không?
- Nguồn: [DDIM](https://arxiv.org/abs/2010.02502), [official DDIM code](https://github.com/ermongroup/ddim), [PNDM](https://arxiv.org/abs/2202.09778), [DPM-Solver](https://arxiv.org/abs/2206.00927).

## A2.2 File/idea dự kiến

- `src/diffusion/samplers/ddim.py`: DDIM update và timestep subsequence.
- `src/diffusion/samplers/ode.py`: optional solver interfaces; không chứa training.
- Cùng prediction model/checkpoint với Phase 1 để sampler comparison hợp lệ.

## A2.3 Checklist

- [ ] Định nghĩa inference timesteps giảm dần, không duplicate/mất endpoint.
- [ ] Implement DDIM `x_0` estimate và update equation.
- [ ] Hỗ trợ `eta=0` deterministic và `eta>0` stochastic.
- [ ] Test `eta=0` cùng initial noise cho bitwise/near deterministic output.
- [ ] Test shape/device/dtype ở batch > 1.
- [ ] So sánh 1000/250/100/50/20 bước hoặc tập bước phù hợp config.
- [ ] Đếm NFE thay vì chỉ ghi wall-clock.
- [ ] Dùng cùng initial noise cho DDPM và DDIM grid.
- [ ] Ghi trade-off quality/diversity/speed.
- [ ] Chỉ sau DDIM pass mới chọn một: PNDM hoặc DPM-Solver.
- [ ] Nếu implement solver optional, thêm convergence test trên ODE toy biết nghiệm.

## A2.4 Output và gate

```text
outputs/a2_fast_sampling/<run_id>/
├── same_noise_sampler_grid.png
├── step_count_comparison.png
├── speed_quality.csv
└── summary.md
```

- [ ] DDIM equation/unit/determinism tests pass.
- [ ] Một DDPM checkpoint được sample bằng ít nhất ba step budgets.
- [ ] NFE và wall-clock được báo cáo tách biệt.
- [ ] Optional solver không được đánh dấu done nếu chỉ gọi implementation thư viện.

---

# A3 — Polygon/Mask Spatial Control

## A3.1 Dependency và câu hỏi

- Dependency: Phase 3 text-conditioned base model đã hoạt động; Phase 0 giữ mask trace đúng.
- Câu hỏi: mask/edge map có điều khiển vị trí và silhouette khi prompt giữ nguyên không?
- Nguồn kiến trúc: [ControlNet paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html), [official code](https://github.com/lllyasviel/ControlNet).

## A3.2 Scope

ControlNet gốc dựa vào base model mạnh đã train. Scratch-compatible experiment của project sẽ freeze chính base model do project train rồi học control branch; không tải Stable Diffusion weights.

## A3.3 Checklist

- [ ] Phase 0 export binary mask cùng RGB crop và trace metadata.
- [ ] Tạo edge map deterministic từ mask; không cần pretrained edge detector.
- [ ] Định nghĩa paired input: noisy image, timestep, text, control map, clean target.
- [ ] Kiểm tra control map pixel-aligned sau crop/letterbox/augmentation.
- [ ] Snapshot base checkpoint và freeze đúng parameter.
- [ ] Implement control encoder nhỏ hoặc trainable copy theo config.
- [ ] Implement zero-initialized 1×1 connections.
- [ ] Test zero connections làm initial controlled model gần base output.
- [ ] Test gradient chỉ vào parameter được phép train.
- [ ] Overfit một control-target pair.
- [ ] Overfit mini-batch có mask khác nhau.
- [ ] Same prompt/seed, different mask comparison.
- [ ] Same mask/seed, different prompt comparison.
- [ ] Đo mask alignment bằng IoU nếu có cách chuyển output thành foreground mask; nếu không, báo qualitative protocol rõ.

## A3.4 Output và gate

- [ ] Alignment/zero-init/freeze tests pass.
- [ ] Control branch overfit tiny paired data.
- [ ] Thay mask tạo thay đổi không gian có hệ thống.
- [ ] Base-vs-control degradation và failure cases được báo cáo.

---

# A4 — Noise-Conditional Score Network

## A4.1 Dependency và câu hỏi

- Dependency: F0 analytic Gaussian score pass; Phase 1 giúp hiểu noise-conditioned network.
- Câu hỏi: network có học `s_theta(x, sigma) ≈ ∇x log p_sigma(x)` và Langevin dynamics có phục hồi toy distribution không?
- Nguồn: [NCSN paper](https://arxiv.org/abs/1907.05600), [official code](https://github.com/ermongroup/ncsn).

## A4.2 File/idea dự kiến

- `src/diffusion/score/matching.py`: denoising score matching target/loss.
- `src/diffusion/score/langevin.py`: annealed Langevin sampler.
- Bắt đầu bằng MLP 2D; image U-Net extension là optional.

## A4.3 Checklist

- [ ] Tạo geometric noise-level schedule `sigma_1...sigma_L`.
- [ ] Perturb clean 2D points bằng noise level theo batch.
- [ ] Derive conditional denoising score target.
- [ ] Implement weighted denoising score matching loss.
- [ ] Test analytic Gaussian target trên synthetic Gaussian.
- [ ] Network nhận continuous/discrete noise embedding rõ ràng.
- [ ] Overfit Gaussian score field nhỏ.
- [ ] Vẽ predicted vs analytic vectors.
- [ ] Implement one Langevin update có generator truyền vào.
- [ ] Implement annealed Langevin loop.
- [ ] Sample Gaussian mixture và so sánh mode coverage/statistics.
- [ ] Nối epsilon prediction và score bằng công thức/experiment, không đồng nhất hai tensor thiếu scale factor.
- [ ] Optional: chạy NCSN trên image 32×32 sau toy gate.

## A4.4 Output và gate

- [ ] Analytic score and finite-difference tests pass.
- [ ] Predicted field đúng hướng trên Gaussian toy.
- [ ] Langevin samples phủ các mode chính, có statistics/plot.
- [ ] Phân biệt score objective với DDPM epsilon objective bằng công thức và code names.

---

# A5 — Score SDE and Probability-Flow ODE

## A5.1 Dependency và câu hỏi

- Dependency: A4 đạt gate; F0 Euler integrator test pass.
- Câu hỏi: discrete DDPM/NCSN được biểu diễn trong continuous time thế nào?
- Nguồn: [Score SDE paper](https://openreview.net/forum?id=PxTIG12RRHS), [official JAX code](https://github.com/yang-song/score_sde), [PyTorch code](https://github.com/yang-song/score_sde_pytorch).

## A5.2 Checklist

- [ ] Định nghĩa interface `sde(x,t) -> drift,diffusion`.
- [ ] Implement VP-SDE marginal mean/std.
- [ ] Test marginal statistics bằng Monte Carlo.
- [ ] Map discrete DDPM beta schedule sang VP-SDE intuition.
- [ ] Train continuous-time score model trên 2D toy.
- [ ] Implement reverse-time SDE drift dùng learned score.
- [ ] Implement Euler–Maruyama với Brownian noise đúng scale `sqrt(dt)`.
- [ ] Fixed generator cho stochastic reproducibility.
- [ ] Optional: predictor-corrector sampler.
- [ ] Implement probability-flow ODE drift.
- [ ] Sample cùng initial distribution bằng reverse SDE và ODE.
- [ ] So sánh stochastic diversity, trajectory và numerical error.
- [ ] Optional image 32×32 chỉ sau toy reverse process pass.

## A5.3 Output và gate

- [ ] VP-SDE marginal numerical test pass.
- [ ] Reverse SDE và probability-flow ODE khôi phục toy distribution hợp lý.
- [ ] Step-size ablation cho numerical solver.
- [ ] Có diagram/công thức nối DDPM, NCSN và SDE.

---

# A6 — Autoencoder and Variational Autoencoder

## A6.1 Dependency và câu hỏi

- Dependency: Phase 0 processed images; độc lập về code với score track.
- Câu hỏi: learned latent tensor có nén ảnh mà vẫn giữ thông tin cần cho product generation không?
- Nguồn toán VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

## A6.2 File/idea dự kiến

- `src/autoencoding/encoder.py`: image `[B,3,H,W] → latent parameters/tensor`.
- `src/autoencoding/decoder.py`: latent tensor → reconstruction.
- `src/autoencoding/vae.py`: reparameterization và compose modules.
- `src/autoencoding/losses.py`: reconstruction + KL; không chứa trainer CLI.

## A6.3 Checklist

- [ ] Chốt latent spatial shape và compression factor.
- [ ] Implement deterministic convolutional autoencoder trước.
- [ ] Test input/output shape và pixel range.
- [ ] Overfit một ảnh rồi mini-batch.
- [ ] Lưu reconstruction grid và error map.
- [ ] Đo MSE/PSNR; metric perceptual optional phải document dependency.
- [ ] Implement `mu`, `log_var`, reparameterization cho VAE.
- [ ] Unit test zero-variance/seeded reparameterization behavior hợp lý.
- [ ] Implement KL to standard Normal, reduction được document.
- [ ] Theo dõi reconstruction loss và KL riêng.
- [ ] Kiểm tra posterior collapse/latent variance.
- [ ] Tính train latent mean/std cho normalization, không dùng test statistics.
- [ ] Chốt reconstruction gate trước latent diffusion.

## A6.4 Output và gate

- [ ] AE/VAE shape, reparameterization và KL tests pass.
- [ ] Reconstruction giữ silhouette, màu chính và bố cục theo tiêu chí đã chốt.
- [ ] Latent statistics finite và không collapse hoàn toàn.
- [ ] Decoder ceiling/limitations được ghi rõ trước A7.

---

# A7 — Latent Diffusion Model

## A7.1 Dependency và câu hỏi

- Dependency: A6 reconstruction gate; Phase 1 scheduler/trainer contracts.
- Câu hỏi: diffusion trên spatial latent có giảm compute/memory so với pixel DDPM mà vẫn giữ cấu trúc không?
- Nguồn: [LDM paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html), [official code](https://github.com/CompVis/latent-diffusion).

## A7.2 Checklist

- [ ] Freeze autoencoder cho baseline và test parameter không có gradient.
- [ ] Quyết định sample từ posterior hay dùng posterior mean; document rõ.
- [ ] Normalize latent bằng train statistics.
- [ ] Reuse scheduler với arbitrary channel/spatial shape.
- [ ] U-Net latent input/output channel đúng.
- [ ] Overfit một latent sample và decode trajectory.
- [ ] Overfit mini-batch latent.
- [ ] Train unconditional latent diffusion.
- [ ] Optional: class/text conditioning sau unconditional gate.
- [ ] Decode fixed-seed samples bằng đúng autoencoder checkpoint.
- [ ] So sánh parameter count, peak memory, step time và sample quality với pixel DDPM.
- [ ] Tách reconstruction error khỏi diffusion generation error.
- [ ] Không claim high-resolution benefit nếu chỉ thử 64×64.

## A7.3 Output và gate

- [ ] Latent normalization/shape/freeze tests pass.
- [ ] Latent DDPM tiny overfit và full run tạo sample có cấu trúc.
- [ ] Pixel-vs-latent comparison dùng data/budget được document.
- [ ] Reconstruction ceiling xuất hiện trong report.

---

# A8 — Mini Diffusion Transformer

## A8.1 Dependency và câu hỏi

- Dependency: Phase 1; khuyên dùng sau A7 để làm việc trên latent patches.
- Câu hỏi: Transformer backbone có thay U-Net dưới cùng diffusion objective không?
- Nguồn: [DiT paper](https://arxiv.org/abs/2212.09748), [official code](https://github.com/facebookresearch/DiT).

## A8.2 File/idea dự kiến

- `src/dit/patching.py`: patchify/unpatchify pure transforms.
- `src/dit/blocks.py`: self-attention, MLP, adaLN/modulation.
- `src/dit/model.py`: token sequence → predicted noise/velocity tensor.
- Reuse attention primitives only khi contracts phù hợp; không dùng `nn.MultiheadAttention`.

## A8.3 Checklist

- [ ] Chốt image/latent size chia hết patch size.
- [ ] Implement patchify/unpatchify và exact round-trip test.
- [ ] Implement patch projection và positional embedding.
- [ ] Implement multi-head self-attention bằng Linear/matmul/softmax.
- [ ] Attention probability, mask và gradient tests.
- [ ] Implement MLP, residual và normalization.
- [ ] Implement timestep embedding.
- [ ] Implement class/text condition modulation hoặc adaLN-Zero nhỏ.
- [ ] Zero-init gates/output theo design đã ghi rõ.
- [ ] Unpatchify output về đúng noise tensor shape.
- [ ] Overfit một sample rồi mini-batch.
- [ ] Train mini-DiT với cùng dataset/objective/schedule như U-Net comparison.
- [ ] Match parameter count hoặc report khác biệt minh bạch.
- [ ] So sánh throughput, memory, convergence và fixed-seed samples.

## A8.4 Output và gate

- [ ] Patching/attention/model shape/gradient tests pass.
- [ ] Mini-DiT overfit tiny data.
- [ ] Full run tạo non-noise samples.
- [ ] U-Net/DiT comparison không quy kết architecture khi budgets khác mà không ghi chú.

---

# A9 — Consistency Models

## A9.1 Dependency và câu hỏi

- Dependency: Phase 1 teacher đủ ổn; A5 giúp hiểu probability-flow ODE; A1 EMA hữu ích.
- Câu hỏi: student có học mapping nhất quán để sample một/few steps không?
- Nguồn: [Consistency Models](https://proceedings.mlr.press/v202/song23a.html), [official code](https://github.com/openai/consistency_models).

## A9.2 Scope

Baseline dùng consistency distillation từ teacher do project train. Standalone consistency training là extension riêng; không dùng online pretrained teacher.

## A9.3 Checklist

- [ ] Freeze/version teacher checkpoint và config.
- [ ] Định nghĩa noise/time parameterization tương thích teacher.
- [ ] Implement consistency function với boundary condition tại minimum time.
- [ ] Tạo adjacent-time training pairs bằng teacher/solver.
- [ ] Stop-gradient target branch đúng vị trí.
- [ ] Dùng EMA target network nếu design yêu cầu; test update.
- [ ] Unit test boundary condition gần clean endpoint.
- [ ] Overfit tiny distillation pairs.
- [ ] Train student và sample 1, 2, 4, 8 steps.
- [ ] So sánh teacher many-step với student one/few-step bằng cùng seeds.
- [ ] Báo NFE, latency, diversity và degradation.
- [ ] Optional standalone consistency training là run/config khác.

## A9.4 Output và gate

- [ ] Boundary/teacher-freeze/EMA/pair-generation tests pass.
- [ ] Student tiny overfit thành công.
- [ ] Có sample hợp lệ ở ít nhất one-step và few-step settings.
- [ ] Speed-quality trade-off so với teacher được báo cáo.

---

# A10 — Flow Matching and Rectified Flow

## A10.1 Dependency và câu hỏi

- Dependency: F0 ODE test; khuyên dùng sau A5.
- Câu hỏi: model có regress velocity field rồi vận chuyển noise distribution đến data distribution bằng ODE không?
- Nguồn: [Flow Matching](https://arxiv.org/abs/2210.02747), [official guide/code](https://github.com/facebookresearch/flow_matching), [Rectified Flow](https://arxiv.org/abs/2209.03003), [official code](https://github.com/gnobitab/RectifiedFlow).

## A10.2 File/idea dự kiến

- `src/flows/paths.py`: sample endpoints/interpolation và analytic conditional velocity.
- `src/flows/objective.py`: vector-field regression loss.
- `src/flows/solvers.py`: Euler/Heun integration với trajectory output.

## A10.3 Checklist toy Flow Matching

- [ ] Chốt convention: `t=0` noise, `t=1` data; dùng thống nhất.
- [ ] Sample noise/data endpoint pairs.
- [ ] Implement linear interpolation `x_t=(1-t)x_0+t x_1`.
- [ ] Implement target conditional velocity `x_1-x_0`.
- [ ] Test endpoints và velocity bằng fixture.
- [ ] Train MLP vector field trên Gaussian mixture 2D.
- [ ] Implement Euler ODE sampler.
- [ ] Vẽ trajectories và generated distribution.
- [ ] Step-size/solver ablation.

## A10.4 Checklist image/Rectified Flow

- [ ] Reuse U-Net hoặc mini-DiT với continuous time input.
- [ ] Output velocity cùng shape data/latent.
- [ ] Overfit một image/latent pair rồi mini-batch.
- [ ] Train image/latent flow baseline.
- [ ] Decode/sample bằng Euler/Heun và fixed initial noise.
- [ ] Phân biệt flow velocity loss với DDPM epsilon loss trong code/config.
- [ ] Measure trajectory straightness trên toy data.
- [ ] Optional reflow: generate coupling pairs từ first flow.
- [ ] Optional distillation/few-step run tách riêng.
- [ ] So sánh flow và DDPM chỉ khi data/backbone/budget đủ tương thích.

## A10.5 Output và gate

- [ ] Path/velocity/solver tests pass.
- [ ] Toy flow phục hồi distribution và có trajectory plot.
- [ ] Image flow tiny overfit và full run tạo non-noise samples.
- [ ] Rectified/reflow claims chỉ được ghi khi trajectory/experiment tương ứng tồn tại.

---

# A11 — SD3 and FLUX Architecture Study

## A11.1 Dependency và phạm vi

- Dependency kiến thức: A7 latent, A8 DiT và A10 rectified flow.
- Đây là reading/architecture checkpoint, không train model billion-parameter.
- Nguồn: [SD3 technical paper](https://arxiv.org/abs/2403.03206), [SD3 model card](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX official code](https://github.com/black-forest-labs/flux), [FLUX.1-schnell model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell).

## A11.2 Checklist

- [ ] Vẽ pipeline image ↔ latent ↔ rectified-flow Transformer ↔ text representations.
- [ ] Phân loại từng thành phần theo formulation/representation/backbone/conditioning/solver.
- [ ] Giải thích MMDiT khác pooled condition và U-Net cross-attention baseline thế nào.
- [ ] Ghi rõ SD3 dùng các text encoder pretrained; vì sao trái scratch scope.
- [ ] Ghi rõ FLUX.1-schnell là model 12B/distilled product, không phải một loss duy nhất.
- [ ] Map mỗi block lớn về checkpoint project đã tự xây.
- [ ] Liệt kê phần không thể tái tạo với data/hardware hiện tại.
- [ ] Không chạy/download weights nếu không có task benchmark riêng.

## A11.3 Definition of Done

- [ ] Có architecture note/diagram được kiểm tra link nguồn.
- [ ] Không gọi DDIM/LDM/DiT/ControlNet/FLUX là cùng một trục phân loại.
- [ ] Có bảng “project component → large-model analogue → scale gap”.

---

# Phase 4 — Image Captioning From Scratch

## 4.1 Mục tiêu

Xây model độc lập:

```text
image -> CNN visual tokens -> Transformer decoder -> caption
```

Không dùng pretrained CNN, pretrained tokenizer hoặc `nn.Transformer*`.

## 4.2 Chuẩn bị dataset

- [ ] Chọn Flickr8k làm dataset baseline.
- [ ] Ghi nguồn/license/download instruction.
- [ ] Đặt raw data dưới `data/raw/captioning/flickr8k/`.
- [ ] Xác nhận official train/valid/test split hoặc tạo split không leakage.
- [ ] Thống kê ảnh, caption/ảnh, caption length.
- [ ] Build vocabulary chỉ từ train captions.
- [ ] Chốt image size baseline: `128×128` trước.

## 4.3 File và idea dự kiến

### `configs/phase4_captioning.yaml`

Chứa image size, vocabulary parameters, max length, CNN channels, `d_model`, heads, decoder layers, dropout, optimizer và decoding settings.

### `src/data/caption_dataset.py`

**Idea:** map image/caption pairs thành image tensor + shifted token sequence; collate padding theo batch.

- `CaptionDataset.__getitem__` trả image và một caption token sequence.
- `caption_collate_fn` trả `[B,3,H,W]`, `[B,L]`, padding mask.

### `src/captioning/cnn_encoder.py`

**Idea:** CNN random-init giữ spatial feature map, không global pool.

- `CNNEncoder.forward(images) -> visual_tokens`.
- Ví dụ output `[B, H_v*W_v, D]`.
- Thêm/project visual positional encoding.

### `src/captioning/decoder.py`

**Idea:** decoder stack tự triển khai.

Mỗi block:

1. Masked multi-head self-attention.
2. Add + LayerNorm.
3. Cross-attention với visual tokens.
4. Add + LayerNorm.
5. Feed-forward.
6. Add + LayerNorm.

### `src/captioning/model.py`

**Idea:** compose encoder + decoder + vocabulary head; không chứa training loop.

- Input teacher forcing `[B,L]`.
- Output logits `[B,L,V]`.

### `src/captioning/decoding.py`

**Idea:** inference autoregressive tách khỏi model forward.

- `greedy_decode`.
- `beam_search_decode` sau khi greedy đúng.

### `src/captioning/trainer.py`

- Shift-right input/target.
- Cross-entropy ignore PAD.
- Train/validation metrics.

### CLI/tests

- `scripts/prepare_captioning.py`.
- `scripts/train_phase4.py`.
- `scripts/caption_image.py`.
- `tests/test_caption_dataset.py`.
- `tests/test_caption_masks.py`.
- `tests/test_caption_model.py`.
- `tests/test_decoding.py`.

## 4.4 Checklist triển khai

### A. Data/text

- [ ] Parse Flickr8k captions.
- [ ] Validate every caption maps to an image.
- [ ] Reuse word-level tokenizer primitives từ Phase 3 khi phù hợp.
- [ ] Build vocabulary train-only.
- [ ] Implement caption length policy.
- [ ] Implement image normalization từ thống kê đã chọn/document.
- [ ] Implement collate/padding mask.
- [ ] Inspect one batch và báo shape/range.

### B. CNN encoder

- [ ] Thiết kế resolution/channel table.
- [ ] Implement conv blocks random-init.
- [ ] Giữ spatial tokens.
- [ ] Project channels về `d_model`.
- [ ] Thêm visual positional encoding.
- [ ] Shape test nhiều batch size.
- [ ] Gradient test từ caption loss về CNN đầu tiên.

### C. Transformer decoder

- [ ] Reuse/self-implement attention primitives theo rule.
- [ ] Implement causal mask `[L,L]`.
- [ ] Test token position không attend future positions.
- [ ] Implement token embedding và text positional encoding.
- [ ] Implement decoder self-attention.
- [ ] Implement cross-attention với visual tokens.
- [ ] Implement feed-forward, residual và LayerNorm.
- [ ] Stack N decoder layers.
- [ ] Implement vocabulary projection `[B,L,D] -> [B,L,V]`.
- [ ] Test logits shape và finite gradient.

### D. Training

- [ ] Input: `<BOS> token_1 ... token_n`.
- [ ] Target: `token_1 ... token_n <EOS>`.
- [ ] Cross-entropy ignore PAD.
- [ ] Overfit 1 image-caption pair.
- [ ] Overfit 10 images/multiple captions.
- [ ] Kiểm tra model không chỉ sinh caption phổ biến nhất.
- [ ] Train Flickr8k full train split.
- [ ] Validation bằng teacher-forcing loss/perplexity.
- [ ] Checkpoint/resume smoke test.

### E. Decoding

- [ ] Implement greedy decode bắt đầu BOS.
- [ ] Stop từng sample tại EOS hoặc max length.
- [ ] Không phát PAD/BOS vào câu output cuối.
- [ ] Deterministic greedy test bằng mock logits.
- [ ] Implement beam search sau greedy.
- [ ] Length penalty policy rõ ràng.
- [ ] Test beam bookkeeping và EOS termination.

### F. Evaluation

- [ ] Implement BLEU-1 đến BLEU-4 hoặc xác minh implementation bằng fixture biết trước.
- [ ] Tính metric trên test split, dùng nhiều reference captions.
- [ ] Lưu predictions JSON.
- [ ] Lưu qualitative samples: tốt, trung bình, sai.
- [ ] Phân tích repetition, hallucination và generic caption.
- [ ] Optional: visualize cross-attention theo generated token.

## 4.5 Output bắt buộc

```text
outputs/phase4_captioning/<run_id>/
├── vocabulary.json
├── metrics.json
├── predictions.json
├── loss_curve.png
├── qualitative_samples/
└── attention_maps/
```

## 4.6 Definition of Done

- [ ] Data/tokenizer/mask/attention/model/decoding tests pass.
- [ ] Gradient từ loss đi tới CNN encoder.
- [ ] Model overfit được tiny dataset.
- [ ] Full training có validation curve và checkpoint.
- [ ] Greedy decoding sinh câu kết thúc hợp lệ.
- [ ] BLEU-1..4 được tính đúng trên test.
- [ ] Có phân tích qualitative và failure cases.

---

# Phase 5 — Evaluation, Ablation, Integration and Report

## 5.1 Mục tiêu

Biến các phase thành một nghiên cứu có thể tái lập, không nhất thiết ghép thành một model duy nhất.

## 5.2 File và idea dự kiến

### `src/metrics/diffusion.py`

- Diversity giữa seed.
- Pixel/color distribution diagnostics.
- Class consistency nếu có classifier from-scratch.

### `src/metrics/memorization.py`

- Nearest training image theo pixel feature hoặc feature extractor tự train.
- Lưu nearest-neighbor panels để review, không chỉ một scalar.

### `src/metrics/bleu.py`

- N-gram counting, clipped precision, brevity penalty và BLEU aggregation.
- Fixture test với ví dụ biết trước.

### `scripts/evaluate_diffusion.py`

Checkpoint/config/fixed seeds → sample set + metrics + nearest neighbors.

### `scripts/evaluate_captioning.py`

Checkpoint/test metadata → predictions + BLEU + qualitative selection.

### `scripts/demo.py` hoặc notebook cuối

Chỉ gọi API nội bộ đã test. Không duplicate preprocessing/model code.

## 5.3 Checklist evaluation

### Diffusion

- [ ] Chọn checkpoint bằng tiêu chí định trước, không nhìn test rồi chọn.
- [ ] Generate fixed number samples/class/prompt.
- [ ] Cùng seed khác condition.
- [ ] Cùng condition khác seed.
- [ ] Denoising trajectory.
- [ ] Diversity diagnostic.
- [ ] Memorization nearest-neighbor panel.
- [ ] Failure cases: noise, collapse, artifact, class bleed.

### Captioning

- [ ] Chọn checkpoint bằng validation metric.
- [ ] BLEU-1..4 trên test.
- [ ] Perplexity hoặc test loss.
- [ ] Greedy vs beam comparison.
- [ ] Good/average/bad examples.
- [ ] Failure cases: repetition, missing objects, hallucination.

### Advanced diffusion checkpoints đã thực hiện

- [ ] Sampler comparison báo cả NFE và wall-clock.
- [ ] Score/flow toy distributions có plot và numerical diagnostics.
- [ ] Pixel-vs-latent comparison tách reconstruction error khỏi generation error.
- [ ] U-Net-vs-DiT comparison ghi parameter/compute mismatch.
- [ ] One/few-step comparison báo quality/diversity degradation so với teacher.
- [ ] Mọi checkpoint deferred được ghi rõ, không tính vào kết quả hoàn thành.

## 5.4 Checklist ablation

Chọn ít nhất 2 ablation có câu hỏi rõ ràng, không chạy chỉ để tăng số bảng.

- [ ] `32×32` vs `64×64` hoặc ghi lý do không chạy.
- [ ] Linear vs cosine beta schedule.
- [ ] U-Net không attention vs có middle attention.
- [ ] Background trắng vs neutral random.
- [ ] Condition guidance scale.
- [ ] Caption decoder 1 layer vs 2 layers.
- [ ] Greedy vs beam search.

Mỗi ablation phải giữ các biến còn lại càng giống càng tốt và dùng cùng seed khi phù hợp.

## 5.5 Reproducibility checklist

- [ ] Mỗi run có config snapshot.
- [ ] Seed Python/NumPy/PyTorch.
- [ ] Ghi Python/PyTorch/CUDA versions.
- [ ] Ghi GPU/CPU và thời gian train.
- [ ] Checkpoint resume hoạt động.
- [ ] Raw dataset download/setup được document.
- [ ] Processed dataset sinh lại bằng command.
- [ ] Train/evaluate/sample command được document.
- [ ] Không cần pretrained weights.

## 5.6 README/report checklist

- [ ] Problem statement.
- [ ] Scope và out-of-scope.
- [ ] Dataset và license.
- [ ] Phase architecture diagrams.
- [ ] DDPM math nối với code tensor.
- [ ] U-Net architecture table.
- [ ] Conditioning mechanism.
- [ ] Captioning architecture.
- [ ] Training setup.
- [ ] Metrics và protocol.
- [ ] Results và ablation.
- [ ] Failure cases.
- [ ] Limitations.
- [ ] Reproduce commands.
- [ ] Future work.
- [ ] Taxonomy phân biệt formulation, representation, backbone, condition và sampler.
- [ ] Paper/code/model links trỏ tới nguồn gốc hoặc official repository/model card.
- [ ] Model lớn chỉ được dùng làm architecture study nếu không thuộc scratch experiment.

## 5.7 Definition of Done

- [ ] Test suite pass trong môi trường được hỗ trợ.
- [ ] Phase 1–4 đều có checkpoint/artifact/metric theo scope thực hiện.
- [ ] Ít nhất hai ablation được báo cáo trung thực.
- [ ] Có memorization/failure-case analysis.
- [ ] README đủ để người khác setup, train và evaluate.
- [ ] Demo/notebook chỉ gọi implementation trong `src/`.

---

# 6. Checklist chung cho mỗi pull/change

## Trước khi code

- [ ] Đã đọc `AGENTS.md` và phase tương ứng trong `task.md`.
- [ ] Đã kiểm tra `git status` và không ghi đè thay đổi người dùng.
- [ ] Đã mô tả vấn đề, file owner và contract.
- [ ] Đã xác định verification nhỏ nhất.

## Trong khi code

- [ ] Logic chính nằm trong `src/`, CLI mỏng.
- [ ] Public APIs có type hints/docstring.
- [ ] Tensor shapes/ranges được document.
- [ ] Không hardcode absolute path/hyperparameter thí nghiệm.
- [ ] Không dùng pretrained/high-level forbidden implementations.
- [ ] Không trộn code phase sau.

## Sau khi code

- [ ] Relevant tests pass.
- [ ] Forward/backward smoke test pass nếu có model.
- [ ] Không NaN/Inf.
- [ ] Artifact QA được inspect nếu có xử lý ảnh.
- [ ] Command và kết quả được báo cáo.
- [ ] Chỉ checkbox có evidence mới được đánh dấu `[x]`.
- [ ] Bước tiếp theo nhỏ nhất đã được đề xuất.

# 7. Mốc tiếp theo hiện tại

Checkpoint active tiếp theo là **F0 — Math and Toy Distributions**. Sau numerical gate đầu tiên, thực hiện song song phần không phụ thuộc model của **Phase 0 — Product Data Preparation**.

Vertical slice code đầu tiên của data nên là:

```text
đọc một COCO annotation
→ tạo một binary mask
→ crop một instance
→ letterbox 64×64
→ lưu/hiển thị một ảnh QA
→ viết test cho shape và bounds
```

Không batch-process toàn bộ dataset trước khi vertical slice một instance được kiểm chứng trực quan.

Vertical slice F0 đầu tiên:

```text
sample Gaussian mixture 2D
→ kiểm tra empirical mean/covariance
→ thêm noise tại nhiều timestep
→ vẽ forward trajectory
→ so sánh iterative và closed-form statistics
```
