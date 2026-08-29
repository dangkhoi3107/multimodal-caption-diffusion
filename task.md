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
- [x] Phase 0 — Product data preparation.
- [x] Phase 1 — Unconditional pixel DDPM. **Phase gate closed: full regression test pass; training, sampling và required evidence đã được kiểm chứng.**
- [ ] A1 — Improved DDPM/EDM design experiments.
- [ ] A2 — DDIM and optional fast solvers.
- [x] Phase 2 — Class-conditioned DDPM + CFG. **Phase gate closed: 3-class scratch training, CFG sanity, same-noise class control, diversity và memorization sanity check đã được kiểm chứng.**
- [x] Phase 3 — Text-conditioned DDPM. **Phase gate closed for pooled-text functional baseline: scratch text encoder, prompt dropout, full training, text CFG và same-noise color prompt swaps đã được kiểm chứng; compositional generalization vẫn được ghi rõ là hạn chế.**
- [ ] A3 — Mask/edge spatial control.
- [ ] A4 — NCSN/score matching.
- [ ] A5 — Score SDE/probability-flow ODE.
- [ ] A6 — Autoencoder/VAE.
- [ ] A7 — Latent diffusion.
- [ ] A8 — Mini-DiT.
- [ ] A9 — Consistency models.
- [ ] A10 — Flow Matching/Rectified Flow.
- [ ] A11 — SD3/FLUX architecture study.
- [x] Phase 4 — Image captioning. **Functional scratch baseline closed under an explicit scope exception: product captions thay Flickr8k; tiny-overfit/multi-reference/attention visualization vẫn deferred; beam decoding và systematic caption failure analysis đã được bổ sung ở Phase 5.**
- [ ] Phase 5 — Evaluation, ablation and report.

## 2. Trạng thái repository hiện tại

- [x] Repository đã có skeleton `src/captioning`, `src/diffusion`, `src/utils`.
- [x] Documentation map và ownership skeleton cho sampler/score/consistency/autoencoding/DiT/flow đã được tạo.
- [x] Dataset sản phẩm COCO được giữ immutable dưới `data/raw/products/`.
- [x] Phase 1 giữ dataset một SKU Lifebuoy với `97` train / `31` valid / `16` test instances sau preprocessing.
- [x] Phase 2 đã mở rộng lên `3` SKU classes với `456` train / `147` valid / `80` test instances, tổng `683` instances.
- [x] Phase 2 class mapping ổn định: `0=dove_body_serum_glow_recharge_547ml`, `1=dove_deodorant_niacinamide_omega_40ml`, `2=lifebuoy_handwash_vitamin_protection_400g`; null CFG class là `3`.
- [x] Đã có processed crop datasets `data/processed/products_64/` và `data/processed/products_multiclass_64/` cùng metadata/QA artifacts.
- [x] Đã có unconditional pixel-space DDPM Phase 1: scheduler, timestep embedding, U-Net, trainer và sampler.
- [x] Đã có Phase 2 class conditioning, conditional U-Net, condition dropout, conditional trainer và CFG sampler.
- [x] Phase 2 full scratch training trên Tesla T4 hoàn thành; best checkpoint epoch `99`, best valid loss `0.0144293566`.
- [x] Full regression suite sau Phase 2 pass: `115 passed`.
- [x] Phase 3 có controlled caption metadata cho cùng `456` train / `147` valid / `80` test instances; caption dùng brand/product/color/package quan sát được.
- [x] Phase 3 vocabulary build **chỉ từ train captions** có `19` tokens, special IDs cố định `PAD=0`, `BOS=1`, `EOS=2`, `UNK=3`, `max_length=10`; train/valid/test đều `0` OOV trong controlled schema.
- [x] Đã có tokenizer deterministic, custom scaled-dot-product attention, custom multi-head attention, scratch text encoder, pooled text conditioner, prompt dropout, `TextConditionalUNet`, text trainer và text CFG sampler.
- [x] Phase 3 mini-batch overfit trên `6` samples đạt `final/initial = 0.0554933`; mean prompt-dropout fraction `0.101556`.
- [x] Phase 3 full scratch training trên Tesla T4 hoàn thành với `4,970,051` parameters; best checkpoint epoch `98`, best valid loss `0.0145077739`.
- [x] Full text-CFG fixed-noise evaluation pass: CFG `0` cho pairwise MAD đúng `0`; CFG `1→2→3` làm prompt differences tăng có hệ thống.
- [x] Same-noise color prompt-swap ở CFG `2.0` tạo measurable changes: serum white↔red `0.087546`, deodorant blue↔white `0.067669`, Lifebuoy red↔blue `0.139538` mean absolute difference.
- [x] Phase 3 limitation đã được ghi rõ: color/package/brand vẫn tương quan mạnh với SKU trong train data, nên kết quả mới chứng minh **partial text sensitivity**, không chứng minh full compositional disentanglement.
- [x] Phase 4 đã có `CaptionDataset` teacher-forcing contract, scratch CNN image encoder, causal Transformer decoder, visual cross-attention, end-to-end caption model, training loop, greedy autoregressive generation và BLEU metrics.
- [x] Phase 4 dùng cùng product split `456` train / `147` valid / `80` test, image size `64×64`, train-only vocabulary `19` tokens; model có `4,386,624` parameters.
- [x] Phase 4 full scratch training trên Tesla T4 hoàn thành `100` epochs; checkpoint tốt nhất được chọn **chỉ bằng validation loss** tại epoch `17`, best valid loss `0.1567438867`.
- [x] Phase 4 held-out autoregressive test trên `80` ảnh đạt Exact Match `0.3000`, BLEU-1 `0.8687`, BLEU-2 `0.7397`, BLEU-3 `0.5966`, BLEU-4 `0.4985`.
- [x] Phase 4 scope exception đã được ghi rõ: tiny overfit bị người dùng chủ động bỏ qua để rút thời gian; Flickr8k, multi-reference BLEU và attention visualization vẫn deferred. Beam search + caption failure analysis đã được thực hiện ở Phase 5.
- [x] Phase 5 portfolio integration có Streamlit playground hai chiều; inference adapter restore đúng checkpoint Phase 3/4, letterbox ảnh upload và chỉ gọi implementation trong `src/`.
- [x] Phase 5 caption decoding ablation trên `80` test samples: greedy BLEU-4 `0.4906`, beam-size `3` BLEU-4 `0.5152`; beam tăng BLEU-4 `+0.0246` nhưng exact match giảm `-0.0125`.
- [x] Phase 5 visual-conditioning ablation: real/zero/class-mismatched visual tokens cho BLEU-4 `0.4906/0.2385/0.0000` và controlled class accuracy `1.00/0.35/0.00`.
- [x] Full regression suite sau public-deploy integration: `208 passed` trong Conda environment với pytest temp đặt trong workspace.
- [x] Checkpoint/model weights và experiment artifacts được lưu local/Kaggle, không commit model weights.

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

- [x] Xác nhận đường dẫn raw dataset trong config, không hardcode.
- [x] Ghi lại số ảnh, số annotation và class theo từng split.
- [x] Xác nhận `category_id=0` nếu chỉ là supercategory sẽ không trở thành class train.
- [x] Chốt output size ban đầu: `64×64`.
- [x] Chốt margin quanh bbox: `10%` cho baseline.
- [x] Chốt background baseline: trắng hoàn toàn; random neutral để ablation sau.
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

- [x] Tạo `phase0_data.yaml` với path tương đối.
- [x] Tạo dataclass/type cho image, annotation và category.
- [x] Implement `load_coco`.
- [x] Implement `build_coco_indexes`.
- [x] Báo lỗi rõ khi ảnh hoặc annotation thiếu field.
- [x] Test category mapping và instance count trên fixture nhỏ.

### B. Polygon và crop

- [x] Implement polygon → binary mask.
- [x] Test polygon đơn giản bằng hình chữ nhật biết trước diện tích.
- [x] Hỗ trợ một annotation có nhiều polygon.
- [x] Implement bounds convention và test off-by-one.
- [x] Implement margin và clip vào image bounds.
- [x] Implement masked composite trên nền trắng.
- [x] Implement resize giữ aspect ratio.
- [x] Implement padding vuông đối xứng.
- [x] Xác nhận output `[64, 64, 3]`, RGB, `uint8`.

### C. Dataset generation

- [x] Tạo tên file output deterministic theo split/image/annotation ID.
- [x] Ghi `train.jsonl`, `valid.jsonl`, `test.jsonl`.
- [x] Metadata có `source_image`, `image_id`, `annotation_id`, `class_id`, `class_name`.
- [x] Metadata có thông tin bounds và preprocessing config/version.
- [x] Giữ nguyên split gốc.
- [x] Ghi summary: số processed, skipped và lỗi theo split/class.
- [x] Không ghi bất kỳ output nào vào `data/raw/`.

### D. QA

- [x] Sinh contact sheet tối thiểu 32 crop train.
- [x] Sinh contact sheet cho toàn bộ valid/test nếu số lượng nhỏ.
- [x] Kiểm tra thủ công object không bị cắt nắp/cạnh quan trọng.
- [x] Kiểm tra không kéo méo aspect ratio.
- [x] Kiểm tra mask không giữ vùng background lớn bất thường.
- [x] Tìm duplicate/gần duplicate và ghi nhận, chưa tự ý xóa.
- [x] Load một DataLoader batch.
- [x] Báo cáo batch shape `[B, 3, 64, 64]`, dtype và min/max.
- [x] Test deterministic: chạy hai lần cùng seed/config cho cùng checksum metadata.

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

- [x] Toàn bộ unit test Phase 0 pass.
- [x] Không có output crop sai shape/dtype/range.
- [x] DataLoader trả `[B, 3, 64, 64]` trong `[-1, 1]`.
- [x] Contact sheet đã được người dùng hoặc dev kiểm tra.
- [x] Tất cả crop trace ngược được về raw annotation.
- [x] Dataset processed sinh lại được từ một command và config.

---

# Phase 1 — Unconditional Pixel-space DDPM

## 1.1 Mục tiêu

Xây DDPM không có class/text condition:

```text
Gaussian noise [B, 3, 64, 64] → product-like image [B, 3, 64, 64]
```

Model học dự đoán noise `epsilon` từ `x_t` và timestep `t`.

## 1.2 Kiến thức/công thức phải hiểu trước

- [x] Giải thích `beta_t`, `alpha_t = 1 - beta_t`.
- [x] Giải thích `alpha_bar_t = product(alpha_1 ... alpha_t)`.
- [x] Mapping công thức sang tensor shape `[T]`.
- [x] Giải thích sampling trực tiếp:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

- [x] Giải thích target `epsilon` và MSE loss.
- [x] Giải thích vì sao U-Net cần timestep embedding.
- [x] Giải thích reverse mean, variance và noise tại `t > 0`.

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

- [x] Implement linear beta schedule.
- [x] Assert `0 < beta_t < 1` và monotonic cho config baseline.
- [x] Precompute alphas, cumulative products và posterior coefficients.
- [x] Implement batch-safe `extract`.
- [x] Implement `q_sample` với noise truyền vào để test deterministic.
- [x] Test `q_sample` shape/device/dtype.
- [x] Visualize cùng một ảnh tại `t = 0, 100, 300, 500, 999`.
- [x] Xác nhận timestep cuối gần Gaussian noise.

### B. Timestep embedding

- [x] Implement sin/cos embedding không dùng thư viện model cấp cao.
- [x] Xử lý embedding dimension lẻ.
- [x] Test cùng timestep cho cùng vector.
- [x] Test timestep khác tạo vector khác.
- [x] Implement two-layer MLP.

### C. U-Net blocks

- [x] Implement normalization + activation + conv order đã chọn và document.
- [x] Project time embedding vào channel dimension.
- [x] Implement residual shortcut khi input/output channels khác nhau.
- [x] Test gradient đi qua time projection.
- [x] Implement downsample.
- [x] Implement upsample.
- [x] Test từng block với batch > 1.

### D. Full U-Net

- [x] Vẽ/báo cáo channel và resolution ở từng stage.
- [x] Implement down path và lưu skip tensors.
- [x] Implement middle blocks.
- [x] Implement up path và concatenate/add skip đúng channel.
- [x] Output conv trả đúng 3 channels.
- [x] Test input/output `[2, 3, 64, 64]`.
- [x] Test loss backward tạo finite gradients.
- [x] Báo cáo parameter count.
- [x] Chưa thêm attention nếu baseline chưa pass.

### E. Training correctness ladder

- [x] Implement random timestep per sample.
- [x] Implement random Gaussian noise cùng shape `x_0`.
- [x] Implement epsilon-prediction MSE.
- [x] Overfit **một ảnh**; đặt tiêu chí loss/sample trước khi chạy.
- [x] Lưu fixed-noise sample trong one-image experiment.
- [x] Overfit **một mini-batch 8–16 ảnh**.
- [x] Xác nhận không NaN/Inf ở loss và gradient.
- [x] Chỉ sau đó train toàn train split.
- [x] Validation chỉ đo trên valid split, không update parameter.

### F. Reverse sampling

- [x] Implement reverse mean.
- [x] Implement posterior variance.
- [x] Không thêm random noise khi `t=0`.
- [x] Implement full loop `T-1 → 0`.
- [x] Hỗ trợ fixed `torch.Generator`.
- [x] Test cùng seed/checkpoint tạo cùng output.
- [x] Lưu denoising trajectory có số frame giới hạn.
- [x] Visualization clamp/unnormalize chỉ ở boundary; `predicted_x_0` clipping trong reverse sampler được document như một lựa chọn ổn định hóa thuật toán. Final sample đã được kiểm tra không còn overflow ngoài `[-1, 1]`.

### Bằng chứng Phase 1 hiện có

- Overfit one-image: noise-prediction MSE giảm mạnh trong 1000 steps; không NaN/Inf.
- Overfit mini-batch 8 ảnh: correctness gate pass.
- Full training trên `97` train / `31` valid images bằng CUDA; U-Net có `4,603,587` parameters.
- Best checkpoint ở epoch `98`: train loss `0.0223723`, valid loss `0.0422775`.
- Reverse sampler từng bị positive drift; đã sửa bằng posterior mean từ clipped `predicted_x_0` và regression tests.
- Final fixed-seed sample stats sau fix: min `-0.999834`, max `0.999834`, mean `0.495197`, std `0.575636`, tỷ lệ `< -1` và `> 1` đều bằng `0`.
- Final sample grid đã sinh được nhiều product-like image màu đỏ/nền trắng, không còn pure noise/white collapse.
- Full regression test suite pass sau khi hoàn tất Phase 1.

### Evidence còn thiếu để đóng Phase 1 gate

- [x] Forward-noising visualization của **ảnh thật** tại `t = 0, 100, 300, 500, 999`.
- [x] Báo cáo thống kê/chứng minh `x_999` của ảnh thật gần Gaussian noise.
- [x] Lưu fixed-noise sample riêng cho one-image overfit experiment.
- [x] Xác nhận/snapshot denoising trajectory cuối cùng vào artifact path chuẩn.
- [x] Smoke test resume checkpoint.
- [x] Ghi hardware và git commit cho run baseline; duration không được đo trong run gốc nên được document là unavailable thay vì ước lượng.

### G. Experiment hygiene

- [x] Config lưu cùng run.
- [x] Checkpoint có model, optimizer, epoch/step, config, seed.
- [x] Resume checkpoint được smoke test.
- [x] Loss curve được lưu.
- [x] Sample grid dùng fixed seed qua các epoch.
- [x] Ghi hardware, duration và git commit nếu có.

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

- [x] Scheduler/unit/shape/gradient tests pass.
- [x] Forward diffusion visualization đúng trực quan.
- [x] Model overfit được một ảnh.
- [x] Model overfit được mini-batch.
- [x] Reverse sampler deterministic với fixed seed và không NaN.
- [x] Full dataset training có checkpoint, curve và samples.
- [x] Sample có cấu trúc/màu sắc khác pure noise, không yêu cầu logo đúng.


**Phase 1 status:** COMPLETE — phase gate closed after full regression pass and artifact/evidence verification.
---

# Phase 2 — Class-Conditioned DDPM

## 2.1 Mục tiêu

Mở rộng diffusion model từ unconditional sang class-conditioned:

```text
ConditionalUNet(x_t, timestep, class_id) -> predicted_noise
```

và chứng minh class condition ảnh hưởng kết quả khi giữ nguyên initial noise. Baseline vẫn được train hoàn toàn từ random initialization.

## 2.2 Điều kiện dữ liệu

- [x] Phase 1 đạt Definition of Done.
- [x] Có tối thiểu 2 class; baseline Phase 2 dùng `3` SKU classes.
- [x] Taxonomy cùng cấp độ SKU-level.
- [ ] Mỗi class có target tối thiểu đã ghi rõ **trước** thu thập. Không có bằng chứng pre-registration cho ngưỡng này nên không backfill checkbox.
- [ ] Mọi object thuộc class mục tiêu xuất hiện trong ảnh đã được annotate nhất quán. COCO annotations hợp lệ nhưng chưa có audit độc lập để chứng minh không bỏ sót object.
- [x] Thống kê train/valid/test và imbalance theo class.
- [x] Roboflow exports dùng cho Phase 2 báo không có augmentation; project giữ nguyên split nguồn và không tạo augmented clone để phân tán qua split.

Class mapping ổn định:

```text
0 -> dove_body_serum_glow_recharge_547ml
1 -> dove_deodorant_niacinamide_omega_40ml
2 -> lifebuoy_handwash_vitamin_protection_400g
3 -> NULL / unconditional CFG condition
```

Processed instance counts:

| Split | Class 0 | Class 1 | Class 2 | Total |
|---|---:|---:|---:|---:|
| train | 156 | 203 | 97 | 456 |
| valid | 54 | 62 | 31 | 147 |
| test | 28 | 36 | 16 | 80 |
| **total** | **238** | **301** | **144** | **683** |

Train imbalance lớn nhất/nhỏ nhất khoảng `203 / 97 ≈ 2.09×`; baseline không dùng weighted sampler vì mức này được chấp nhận cho correctness experiment.

## 2.3 File và implementation thực tế

### `configs/phase2_data.yaml`

Mô tả multi-source preprocessing, source-folder → project class mapping và output `data/processed/products_multiclass_64/`.

### `configs/phase2_class_conditional.yaml`

Kế thừa scheduler/U-Net baseline và thêm `num_classes=3`, `condition_dropout=0.10`, training config và mini-batch overfit config.

### `src/diffusion/conditioning.py`

- `ClassConditioner(num_classes, embedding_dim)`.
- Reserved `null_class_id=num_classes`.
- `drop_condition(class_ids, probability, generator)`.
- Real/null class đều được test shape/range/gradient.

### `src/diffusion/conditional_unet.py`

Giữ `UNet` Phase 1 không đổi để regression tiếp tục chạy. Phase 2 dùng:

```text
time_emb  [B, D]
    +
class_emb [B, D]
    =
condition_emb [B, D]
```

`condition_emb` được đưa vào các residual blocks.

### `src/diffusion/conditional_trainer.py`

- Conditional epsilon-prediction MSE.
- Random timestep/noise per sample.
- Condition dropout cho CFG training.
- Full train/validation loop không sửa Phase 1 trainer.

### `src/diffusion/conditional_sampler.py`

CFG:

```text
eps_cfg = eps_uncond + scale * (eps_cond - eps_uncond)
```

Reverse posterior dùng scheduler Phase 1 đã kiểm chứng.

### Scripts/tests thực tế

- `scripts/inspect_phase2_raw_data.py`
- `scripts/prepare_phase2_products.py`
- `scripts/inspect_phase2_dataset.py`
- `scripts/overfit_phase2_minibatch.py`
- `scripts/train_phase2_conditional.py`
- `scripts/evaluate_phase2_cfg.py`
- `scripts/evaluate_phase2_full_cfg.py`
- `scripts/evaluate_phase2_quality.py`
- `tests/test_conditioning.py`
- `tests/test_conditional_unet.py`
- `tests/test_conditional_trainer.py`
- `tests/test_conditional_sampler.py`

## 2.4 Checklist triển khai

### A. Data expansion

- [x] Chốt danh sách class và mapping ID ổn định.
- [x] Chuẩn hóa project class name theo SKU-level snake_case naming.
- [x] Thu thập/nhập thêm hai SKU class đã được annotate COCO.
- [x] Chạy preprocessing multi-class dựa trên primitive Phase 0.
- [x] Review contact sheet riêng từng class.
- [x] Kiểm tra imbalance; quyết định chưa cần sampler/weight cho baseline.

### B. Class conditioning

- [x] Implement class embedding.
- [x] Implement null class embedding.
- [x] Implement condition dropout ở training.
- [x] Combine class/time embedding và document shape `[B, D]`.
- [x] Test class IDs ngoài range báo lỗi.
- [x] Test null/real condition đều forward được.
- [x] Test gradient tới class embedding.

### C. Training

- [x] Khởi tạo random toàn bộ model; không load Phase 1 weights cho experiment chính.
- [x] Warm-start không được dùng cho baseline Phase 2.
- [x] Overfit balanced mini-batch có ít nhất một sample mỗi class; thực tế dùng `2` sample/class.
- [x] Kiểm tra model không bỏ qua class bằng cùng noise khác class.
- [x] Train full multi-class dataset.
- [x] Log train/valid loss, condition-dropout fraction và sample analysis theo class.

### D. Classifier-free guidance

- [x] Implement hai forward cond/uncond khi sampling.
- [x] Test `scale=0` tương đương unconditional prediction.
- [x] Test `scale=1` tương đương conditional prediction theo công thức chọn.
- [x] Sinh CFG comparison grid; overfit sanity dùng `0,1,3,5`, full model dùng `0,1,2,3`.
- [x] Ghi nhận guidance cao làm condition mạnh/aggressive hơn; final visual QA ưu tiên khoảng `1–2`, giữ `3` để stress-test.

### E. Evaluation

- [x] Cùng seed, khác class grid.
- [x] Cùng class, khác seed grid.
- [x] Kiểm tra class collapse về class phổ biến bằng same-class/different-seed visual QA.
- [ ] Train classifier nhỏ từ scratch nếu cần class-consistency metric. Đây là optional metric, không chặn baseline Phase 2.
- [x] Kiểm tra nearest training crop bằng pixel L1 để phát hiện copy gần như nguyên xi.
- [x] Báo cáo limitation/failure cases theo class và seed.

### Bằng chứng Phase 2

**Multi-class preprocessing**

- `456` train / `147` valid / `80` test instances.
- Tổng `683` instances.
- Preprocessing pass có `0` skipped instances.
- DataLoader contract `[B,3,64,64]`, `float32`, normalized trong `[-1,1]`.
- Contact sheets của cả 3 class được visual QA.

**Balanced mini-batch overfit**

- Batch shape: `(6,3,64,64)`.
- Class IDs: `[0,0,1,1,2,2]`.
- Conditional U-Net: `4,604,611` parameters.
- Initial mean loss: `0.2298858922`.
- Final mean loss: `0.0180983363`.
- Final / initial loss: `0.0787274770`.
- Mean observed condition-dropout fraction: `0.10`.

**Full scratch training**

- Hardware: Tesla T4.
- Train/valid: `456 / 147`.
- `100` epochs, batch size `12`, condition dropout `0.10`.
- Best checkpoint: epoch `99`.
- Epoch 99 train loss: `0.012604`.
- Best valid loss: `0.0144293566`.
- Epoch 100 train/valid: `0.013616 / 0.014681`.
- Full regression after implementation: `115 passed`.

**CFG fixed-noise sanity — full model**

```text
scale=0.0 | 0-1=0.000000 | 0-2=0.000000 | 1-2=0.000000
scale=1.0 | 0-1=0.146767 | 0-2=0.261148 | 1-2=0.219714
scale=2.0 | 0-1=0.225588 | 0-2=0.377089 | 1-2=0.319863
scale=3.0 | 0-1=0.275150 | 0-2=0.429068 | 1-2=0.373027
```

`scale=0` cho output giống hệt giữa các class với cùng stochastic path; class differences tăng có hệ thống khi guidance tăng.

**Same-class / different-seed diversity, CFG=2**

```text
class 0: 0.312451
class 1: 0.352489
class 2: 0.407959
```

Nearest-training-crop pixel L1 nằm trong khoảng `0.120786–0.487825`; không có generated sample nào gần `0`, nên sanity check không cho thấy copy pixel gần như nguyên xi từ train set.

**Visual QA và failure cases**

- Class 1 (Dove deodorant) ổn định nhất qua 4 seed, giữ dạng tube và màu xanh/tím.
- Class 0 giữ identity tốt ở seed `101/202`, nhưng seed `303/404` bị structure degradation mạnh.
- Class 2 giữ pouch/red identity tốt ở seed `101/202/404`; seed `303` có color/shape drift.
- Kết luận Phase 2 là **functional baseline**, không claim class consistency hoàn hảo ở mọi seed.

## 2.5 Output thực tế cần giữ

Không cần ép toàn bộ artifact vào một `<run_id>` mới nếu các script hiện tại đã dùng các path ổn định sau. Các artifact tối thiểu cần giữ local/Kaggle:

```text
data/processed/products_multiclass_64/
├── train/
├── valid/
├── test/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── classes.json
└── preprocessing_summary.json

outputs/phase2_data/
├── contact_sheet_train_class_0.png
├── contact_sheet_train_class_1.png
├── contact_sheet_train_class_2.png
├── ... valid/test contact sheets ...
└── qa_report.json

outputs/phase2_overfit_minibatch/
├── model.pt
├── report.json
└── loss_curve.png

outputs/phase2_conditional/
├── best.pt
├── history.json
└── summary.json

outputs/phase2_full_cfg_evaluation/
└── cfg_class_grid.png

outputs/phase2_quality_evaluation/
└── same_class_different_seed.png
```

Khuyến nghị experiment hygiene: lưu thêm `report.json` trong `phase2_full_cfg_evaluation/` và `phase2_quality_evaluation/` để numerical evidence không chỉ tồn tại trong terminal/chat log. Việc này không yêu cầu retrain.

## 2.6 Definition of Done

- [x] Có từ 2 class trở lên với processed dataset hợp lệ.
- [x] Conditional shape/gradient/unit tests pass.
- [x] Mini-batch multi-class overfit thành công.
- [x] Cùng noise, class khác tạo thay đổi có hệ thống.
- [x] CFG hoạt động đúng các boundary scale đã test.
- [x] Không luôn sinh class phổ biến nhất.
- [x] Có memorization sanity check và failure cases.

**Phase 2 status:** COMPLETE — functional baseline closed after 3-class scratch training, full regression, fixed-noise CFG verification, same-class diversity evaluation, nearest-training-crop sanity check và documented seed-dependent failure cases.

---

# Phase 3 — Text-Conditioned DDPM

## 3.1 Mục tiêu

Chuyển từ class ID sang prompt ngắn và train toàn bộ text-conditioned DDPM từ random initialization:

```text
caption/prompt
→ deterministic tokenizer
→ scratch text encoder
→ pooled text condition
→ U-Net epsilon prediction
→ text classifier-free guidance
→ image
```

Baseline Phase 3 dùng **pooled text vector**. Cross-attention là extension tùy chọn sau khi pooled baseline đã đạt gate.

## 3.2 Điều kiện dữ liệu

- [x] Phase 2 đạt Definition of Done.
- [x] Mỗi processed crop có controlled caption mô tả thuộc tính quan sát được.
- [x] Caption chứa brand/product/color/package thay vì chỉ class ID thuần.
- [x] Controlled vocabulary nhỏ và deterministic; background baseline đồng nhất nên không thêm background token riêng.
- [x] Không gán thuộc tính không quan sát được.
- [x] Vocabulary chỉ build từ train split.

Controlled caption schema hiện tại:

```text
class 0 — dove_body_serum_glow_recharge_547ml
  white / dove / body serum / bottle

class 1 — dove_deodorant_niacinamide_omega_40ml
  blue / dove / deodorant / tube

class 2 — lifebuoy_handwash_vitamin_protection_400g
  red / lifebuoy / handwash / pouch
```

Mỗi class dùng ba template deterministic. Thuộc tính màu/package/brand vẫn tương quan mạnh với SKU, vì vậy Phase 3 baseline **không được diễn giải như bằng chứng full compositional disentanglement**. Prompt-swap chỉ đo mức nhạy của model với token ngoài class.

## 3.3 File/implementation đã có

### Config/data preparation

- `configs/phase3_text_conditional.yaml`.
- `scripts/prepare_phase3_captions.py`.
- `scripts/build_phase3_vocabulary.py`.
- `src/data/text_product_dataset.py`.

### Scratch text stack

- `src/text/vocabulary.py` — vocabulary serialize được, special IDs cố định.
- `src/text/tokenizer.py` — normalization/tokenization/encode/decode deterministic.
- `src/text/attention.py` — scaled dot-product attention + custom multi-head attention.
- `src/text/encoder.py` — token embedding + positional encoding + encoder blocks + masked mean pooling.

Không dùng `nn.MultiheadAttention`, `nn.Transformer*`, pretrained tokenizer, CLIP, T5 hoặc pretrained text encoder.

### Diffusion integration

- `src/diffusion/text_conditioning.py` — pooled-text projection + prompt dropout/null prompt.
- `src/diffusion/text_conditional_unet.py` — time embedding + pooled text condition đưa vào ResBlocks.
- `src/diffusion/text_trainer.py` — epsilon objective, step/epoch training và validation.
- `src/diffusion/text_sampler.py` — DDPM reverse sampler với text classifier-free guidance.

### Training/evaluation scripts

- `scripts/overfit_phase3_minibatch.py`.
- `scripts/train_phase3_text_conditional.py`.
- `scripts/evaluate_phase3_cfg.py` — mini-overfit CFG sanity.
- `scripts/evaluate_phase3_full_cfg.py` — best-checkpoint CFG evaluation.
- `scripts/evaluate_phase3_prompt_swap.py` — same-noise single-color-word swaps.

### Tests

- `tests/test_vocabulary.py`.
- `tests/test_tokenizer.py`.
- `tests/test_attention.py`.
- `tests/test_text_encoder.py`.
- `tests/test_text_conditional_unet.py`.
- `tests/test_text_trainer.py`.
- `tests/test_text_product_dataset.py`.
- `tests/test_text_sampler.py`.

## 3.4 Checklist triển khai

### A. Caption metadata

- [x] Định nghĩa controlled caption schema.
- [x] Sinh caption cho từng crop train/valid/test.
- [x] Review caption-class consistency trên các class/templates đại diện.
- [x] Thống kê caption length và token frequency.
- [x] Chốt `max_length=10` từ controlled-caption distribution; không có truncation trong train/valid/test.

### B. Tokenizer/vocabulary

- [x] Implement normalization.
- [x] Implement word splitting/punctuation policy deterministic.
- [x] Build vocab chỉ từ train captions.
- [x] Thêm special tokens với ID cố định: `PAD=0`, `BOS=1`, `EOS=2`, `UNK=3`.
- [x] Implement encode BOS/EOS/pad/truncate.
- [x] Implement decode và bỏ PAD/BOS/EOS đúng cách.
- [x] Round-trip/unit tests pass.
- [x] Unknown-token behavior được unit-test; controlled valid/test hiện có `0` OOV.

Vocabulary evidence:

```text
vocabulary_size = 19
content_length   = 5..8
max_encoded_len  = 10
train OOV        = 0
valid OOV        = 0
test OOV         = 0
```

### C. Attention/text encoder

- [x] Implement scaled dot-product attention từ tensor ops.
- [x] Test output shape và attention weights sum gần `1`.
- [x] Implement boolean mask broadcast và reject fully-masked query.
- [x] Implement multi-head split/merge từ scratch.
- [x] Implement sinusoidal positional encoding từ scratch.
- [x] Implement feed-forward và encoder blocks.
- [x] Implement padding-aware masked mean pooling.
- [x] Test valid token output/pooled representation không phụ thuộc PAD embedding values.
- [x] Test finite gradients.

### D. Text conditioning

- [x] Project pooled text condition về condition/time dimension.
- [x] Kết hợp time/text embedding trong diffusion ResBlocks.
- [x] Implement null prompt và prompt dropout cho CFG training.
- [x] Overfit controlled mini-batch `6` samples với nhiều prompts.
- [x] Same initial noise + canonical prompts thuộc các SKU khác nhau tạo output differences có hệ thống. Đây là class-level prompt evidence; isolated class-word-only intervention vẫn có thể làm thêm như diagnostic.
- [x] Same initial noise + chỉ đổi **color word** tạo measurable output change.
- [x] Kiểm tra model không hoàn toàn bỏ qua non-class words bằng color prompt-swap.

Mini-batch overfit evidence:

```text
parameters             = 4,970,051
initial mean loss      = 0.21519354
final mean loss        = 0.01194181
final / initial        = 0.05549334
mean prompt dropout    = 0.10155556
```

Full training evidence:

```text
device                  = Tesla T4
train / valid            = 456 / 147
parameters               = 4,970,051
epochs                   = 100
prompt dropout           = 0.10
best epoch               = 98
best valid loss          = 0.0145077739
final train loss         = 0.013736
final valid loss         = 0.014829
```

### E. Optional cross-attention extension

- [ ] Chỉ bắt đầu nếu muốn vượt pooled baseline; **không còn là blocker của Phase 3 functional baseline**.
- [ ] Visual feature `[B, HW, C]` làm query.
- [ ] Text states `[B, L, D]` làm key/value.
- [ ] Padding mask chặn PAD key.
- [ ] Chỉ thêm tại middle block trước.
- [ ] So sánh pooled vs cross-attention bằng cùng config/seed.
- [ ] Lưu attention visualization nếu diễn giải được.

### F. Evaluation

- [x] Same seed/different canonical prompt grid.
- [ ] Same prompt/different seed diversity grid cho Phase 3 riêng biệt.
- [x] Test prompt hoán đổi **một thuộc tính** bằng color-word swap.
- [ ] Sampling test với prompt chứa `<UNK>`.
- [ ] Tách riêng failure-case set cho ignored word/class bleed.
- [ ] Phase 3-specific nearest-training-crop memorization check.

Full text-CFG fixed-noise evidence:

```text
scale=0.0 | 0-1=0.000000 | 0-2=0.000000 | 1-2=0.000000
scale=1.0 | 0-1=0.093604 | 0-2=0.284754 | 1-2=0.225499
scale=2.0 | 0-1=0.137428 | 0-2=0.436380 | 1-2=0.371193
scale=3.0 | 0-1=0.175059 | 0-2=0.506274 | 1-2=0.448027
```

Same-noise single-color-word prompt swaps at `CFG=2.0`:

```text
serum_white_vs_red       = 0.087546
deodorant_blue_vs_white  = 0.067669
lifebuoy_red_vs_blue     = 0.139538
```

Kết luận được phép từ baseline:

- Text condition và text CFG hoạt động.
- Output phản ứng có thể đo được khi chỉ đổi color token dưới fixed stochastic path.
- Kết quả cho thấy **partial compositional sensitivity**.
- Không được tuyên bố model đã disentangle hoàn toàn color/package/class vì train attributes còn class-correlated.

## 3.5 Output/evidence hiện có

```text
outputs/phase3_text_conditional/
├── vocabulary.json
├── vocabulary_report.json
├── caption_report.json
├── best.pt
├── last.pt
├── checkpoint_epoch_*.pt
├── history.json
└── summary.json

outputs/phase3_overfit_minibatch/
├── model.pt
└── report.json

outputs/phase3_cfg_evaluation/
├── cfg_prompt_grid.png
└── report.json

outputs/phase3_full_cfg_evaluation/
├── cfg_prompt_grid.png
└── report.json

outputs/phase3_prompt_swap_evaluation/
├── prompt_swap_grid.png
└── report.json
```

Model/checkpoint artifacts không commit vào Git; code/config/tests và `task.md` được commit.

## 3.6 Definition of Done

- [x] Tokenizer/vocab/attention/text encoder tests pass.
- [x] Text-conditioned mini-batch overfit thành công với gate mạnh (`final/initial = 0.05549`).
- [x] Same-seed canonical class prompts tạo output differences; isolated class-word-only swap được ghi là follow-up diagnostic chứ không được overclaim.
- [x] Ít nhất một thuộc tính ngoài class có ảnh hưởng đo được: color-only swaps đều cho non-zero MAD.
- [x] Prompt CFG hoạt động: `scale=0` loại hoàn toàn prompt effect và differences tăng khi guidance tăng.
- [x] Model chính train từ random initialization, không dùng pretrained text encoder/tokenizer.

**Phase 3 status:** COMPLETE — pooled-text functional baseline closed after controlled captions, train-only vocabulary, scratch attention/text encoder, prompt dropout, 6-sample overfit, 100-epoch full scratch training, fixed-noise CFG verification và single-color-word prompt-swap evaluation.

**Known limitation:** train attributes còn tương quan mạnh với SKU; prompt swaps là out-of-distribution combinations. Vì vậy Phase 3 chứng minh text sensitivity/partial compositionality, không chứng minh full attribute disentanglement. Optional cross-attention, Phase-3-specific diversity, `<UNK>` sampling và memorization diagnostics vẫn là follow-up experiments, không được xem như đã hoàn thành.

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

Xây model độc lập hoàn toàn từ random initialization:

```text
image -> CNN spatial visual tokens -> causal Transformer decoder -> caption
```

Không dùng pretrained CNN, pretrained tokenizer, pretrained language model hoặc `nn.Transformer*` / `nn.MultiheadAttention` cho experiment chính.

**Phase 4 status:** COMPLETE — **functional scratch baseline** đã đóng bằng full end-to-end training và held-out autoregressive evaluation.

Scope exception được người dùng chấp nhận để rút ngắn thời gian:

- Dataset baseline thực tế dùng product images + controlled captions đã chuẩn hóa từ Phase 3 thay vì Flickr8k.
- Tiny-overfit gate được chủ động bỏ qua; không được dùng Phase 4 để claim tiny-overfit evidence.
- Multi-reference BLEU và attention visualization chưa thuộc baseline đã đóng. Beam search và systematic caption failure analysis đã được bổ sung, kiểm chứng ở Phase 5.

## 4.2 Dataset baseline thực tế

- [x] Reuse product image/caption records từ Phase 3 thay cho Flickr8k baseline ban đầu.
- [x] Giữ split không leakage đã chốt: `456` train / `147` valid / `80` test.
- [x] Controlled captions dùng brand/product/color/package; caption content length quan sát `5–8` tokens trong schema hiện tại.
- [x] Reuse word-level tokenizer primitives từ Phase 3.
- [x] Build/reuse vocabulary **chỉ từ train captions**: `19` tokens, `PAD=0`, `BOS=1`, `EOS=2`, `UNK=3`.
- [x] Decoder sequence length baseline: `10` teacher-forcing positions; dataset encode nội bộ `L+1` rồi shift.
- [x] Image size baseline thực tế: `64×64`.
- [x] Image tensor normalize về `[-1,1]` theo `ProductImageDataset` contract.
- [ ] Flickr8k baseline, license/download instruction và official split — **DEFERRED; không được claim đã làm**.

## 4.3 File và implementation thực tế

### `configs/phase4_captioning.yaml`

Baseline config:

```text
image_size = 64
sequence_length = 10
model_dim = 256
CNN base_channels = 64
decoder heads = 4
decoder layers = 3
feedforward_dim = 512
dropout = 0.10
batch_size = 16
learning_rate = 3e-4
epochs = 100
```

### `src/data/caption_dataset.py`

Teacher-forcing sample contract:

```text
image          FloatTensor[3,64,64]
caption        str
input_ids      LongTensor[L]
target_ids     LongTensor[L]
padding_mask   BoolTensor[L]
target_mask    BoolTensor[L]
```

Shift:

```text
full_ids   = [BOS, w1, w2, ..., EOS, PAD, ...]
input_ids  = full_ids[:-1]
target_ids = full_ids[1:]
```

### `src/captioning/image_encoder.py`

Scratch CNN random-init:

```text
[B,3,64,64]
-> CNN downsample
-> [B,256,8,8]
-> flatten spatial
-> [B,64,256]
-> learnable visual positional embedding + LayerNorm
```

Smoke evidence:

```text
Input image: (4, 3, 64, 64)
Image tokens: (4, 64, 256)
Grid size: 8
Num image tokens: 64
Parameters: 2,004,032
Phase 4 image encoder smoke test: PASS
```

### `src/captioning/decoder.py`

Scratch autoregressive Transformer decoder:

1. Token embedding + sinusoidal position encoding.
2. Causal self-attention với padding-aware mask.
3. Visual cross-attention: text query, image-token key/value.
4. Feed-forward + residual + LayerNorm.
5. Stack `3` layers ở full config.
6. Vocabulary projection `[B,L,256] -> [B,L,19]`.

Relevant causal test đã chứng minh thay future tokens không làm thay prefix logits.

### `src/captioning/model.py`

Compose:

```text
images
-> ImageEncoder
-> visual tokens
-> CaptionDecoder(input_ids, padding_mask, visual_tokens)
-> logits [B,L,V]
```

### `src/captioning/training.py`

- Next-token cross-entropy ignore PAD.
- Gradient clipping.
- Teacher-forced token accuracy.
- Train/validation epoch loops.

### `src/captioning/generation.py`

- Greedy autoregressive generation bắt đầu từ `BOS`.
- Encode image một lần rồi reuse visual tokens.
- Sinh từng token từ logits cuối sequence.
- Stop khi mọi sample phát `EOS` hoặc đạt `max_length`.

### `src/captioning/metrics.py`

Scratch single-reference corpus BLEU implementation:

- clipped n-gram precision;
- brevity penalty;
- BLEU-1 → BLEU-4.

### CLI

- `scripts/train_phase4_captioning.py` — full train + best checkpoint theo validation loss.
- `scripts/evaluate_phase4_captioning.py` — held-out greedy generation + Exact Match + BLEU + predictions JSON.
- `scripts/smoke_phase4_dataset.py` — real-data dataset smoke.
- `scripts/smoke_phase4_image_encoder.py` — image encoder smoke.
- `scripts/smoke_phase4_decoder.py` — encoder/decoder shape + attention smoke.

## 4.4 Checklist triển khai

### A. Data/text

- [x] Parse/reuse controlled product captions từ Phase 3 metadata.
- [x] Validate caption record đi cùng processed image record qua dataset load/smoke.
- [x] Reuse word-level tokenizer primitives từ Phase 3.
- [x] Build/reuse vocabulary train-only.
- [x] Implement caption length policy với `sequence_length=10` và `L+1` internal encode.
- [x] Reuse image normalization `[-1,1]` đã document.
- [x] Implement padding mask + target mask.
- [x] Inspect real DataLoader batch; dataset smoke pass.
- [ ] Parse Flickr8k captions — **DEFERRED**.

### B. CNN encoder

- [x] Chốt resolution/channel path `64 -> 32 -> 16 -> 8`.
- [x] Implement conv/residual blocks random-init.
- [x] Giữ spatial tokens, không global pool.
- [x] Project channel về `d_model=256`.
- [x] Thêm learnable visual positional embedding.
- [x] Shape tests nhiều batch context + real-data smoke pass.
- [x] Gradient tới CNN được kiểm chứng qua backward tests và full end-to-end caption training.

### C. Transformer decoder

- [x] Reuse custom `MultiHeadAttention` từ Phase 3; không dùng `nn.MultiheadAttention`.
- [x] Implement causal mask `[B,1,L,L]` kết hợp non-PAD keys.
- [x] Test future-token leakage: future token changes không đổi prefix logits.
- [x] Implement token embedding và sinusoidal text positional encoding.
- [x] Implement decoder causal self-attention.
- [x] Implement visual cross-attention với `64` image tokens.
- [x] Implement feed-forward, residual và LayerNorm.
- [x] Stack N decoder layers; full baseline dùng `3` layers.
- [x] Implement vocabulary projection `[B,L,D] -> [B,L,V]`.
- [x] Logits shape/finite gradient tests pass.

### D. Training

- [x] Input teacher forcing: `<BOS> token_1 ... token_n`.
- [x] Target: `token_1 ... token_n <EOS>`.
- [x] Cross-entropy ignore PAD.
- [ ] Overfit 1 image-caption pair — **INTENTIONALLY SKIPPED by user**.
- [ ] Overfit 10 images/multiple captions — **INTENTIONALLY SKIPPED by user**.
- [ ] Systematic check model không chỉ sinh caption phổ biến nhất — chưa có protocol riêng; inspect predictions mới chỉ là qualitative evidence.
- [x] Train full product train split `456` samples trong `100` epochs trên Tesla T4.
- [x] Validation bằng teacher-forcing loss/token accuracy trên `147` samples.
- [x] Best checkpoint selection dùng validation loss, không dùng test: epoch `17`, valid loss `0.1567438867`.
- [ ] Checkpoint **resume** smoke test — checkpoint save đã hoạt động nhưng resume chưa được kiểm chứng.
- [ ] Train Flickr8k full train split — **DEFERRED / out of current baseline**.

Full training evidence:

```text
Device: cuda
GPU: Tesla T4
Vocabulary size: 19
Train samples: 456
Valid samples: 147
Parameters: 4,386,624

Best epoch: 17
Best valid loss: 0.15674388672218842
```

Observed training behavior:

```text
epoch 17: train_loss=0.153821, valid_loss=0.156744
epoch 50: train_loss=0.064362, valid_loss=0.288403
epoch100: train_loss=0.007382, valid_loss=0.644358
```

Kết luận: sau khoảng epoch `17` model overfit rõ; evaluation bắt buộc dùng `best.pt` tại epoch `17`, không dùng epoch `100`.

### E. Decoding

- [x] Implement greedy decode bắt đầu `BOS`.
- [x] Stop tại `EOS` hoặc `max_length`.
- [x] Decode final sentence bỏ special tokens theo tokenizer contract.
- [x] Held-out evaluation chạy autoregressive generation thực sự; không teacher forcing.
- [ ] Deterministic greedy test bằng mock logits — chưa có fixture riêng.
- [x] Implement beam search phục vụ controlled Phase 5 decoding ablation.
- [x] Length penalty policy: cumulative log probability chia `length ** 0.6` trong baseline beam-size `3`.
- [x] Beam bookkeeping/EOS/determinism tests pass bằng known-path dummy decoder.

### F. Evaluation

- [x] Implement scratch corpus BLEU-1 → BLEU-4.
- [x] Chọn checkpoint bằng validation loss trước khi xem test result.
- [x] Tính metric trên held-out test split `80` samples.
- [x] Lưu predictions cùng reference/prediction/exact-match trong evaluation `report.json`.
- [x] Greedy autoregressive evaluation đạt non-trivial held-out metrics.
- [ ] Cross-check BLEU bằng fixture biết trước/reference implementation — chưa được kiểm chứng riêng.
- [ ] Multi-reference BLEU — current metadata evaluation dùng **single reference per record**.
- [ ] Lưu curated qualitative samples: tốt / trung bình / sai — chuyển Phase 5.
- [x] Systematic caption failure analysis lưu missing/extra/repeated tokens và quality bucket cho toàn bộ `80` test records ở `outputs/phase5_caption_decoding/`.
- [ ] Visualize cross-attention theo generated token — optional, chưa làm.

Held-out evidence:

```text
Checkpoint epoch: 17
Test samples: 80
Exact Match: 0.3000
BLEU-1: 0.8687
BLEU-2: 0.7397
BLEU-3: 0.5966
BLEU-4: 0.4985
```

Representative semantically equivalent but non-exact example:

```text
REF : a dove body serum in a white bottle
PRED: a white dove body serum bottle
```

Do đó Exact Match `0.30` phải được đọc cùng BLEU và qualitative outputs; controlled schema có nhiều template diễn đạt cùng product semantics.

## 4.5 Output/artifact thực tế

Kaggle run tạo:

```text
outputs/phase4_captioning/
├── best.pt
├── last.pt
├── checkpoint_epoch_010.pt
├── checkpoint_epoch_020.pt
├── ...
├── history.json
└── summary.json

outputs/phase4_captioning_evaluation/
└── report.json
```

Recommended retained artifacts:

```text
best.pt
history.json
summary.json
phase4_captioning_evaluation/report.json
```

Model weights/artifacts không bắt buộc commit Git; Save Version/archive trên Kaggle là hợp lệ.

## 4.6 Definition of Done

- [x] Core data/tokenizer/mask/attention/model/generation contracts chạy thành công trên real pipeline.
- [x] Gradient từ caption objective đi qua decoder tới scratch CNN trong full end-to-end training.
- [ ] Tiny-overfit gate — **explicitly skipped**; đây là scope exception, không được claim pass.
- [x] Full training có validation history và best checkpoint.
- [x] Greedy decoding sinh held-out captions autoregressively từ `BOS`.
- [x] BLEU-1..4 được tính trên held-out test split và predictions được lưu.
- [ ] Systematic qualitative/failure-case analysis — **deferred to Phase 5**.

**Phase 4 gate decision:** `COMPLETE — functional scratch baseline with documented exceptions.`

Lý do cho phép đóng baseline dù strict original checklist chưa 100%:

1. Người dùng chủ động quyết định bỏ tiny-overfit để rút ngắn thời gian, đúng rule cho phép exception khi được ghi rõ.
2. Full end-to-end training trên `456` samples và held-out test trên `80` samples cung cấp evidence mạnh hơn về pipeline thực sự hoạt động, nhưng **không thay thế claim tiny-overfit**.
3. Các mục chưa làm được ghi rõ là deferred, không bị đánh dấu giả là hoàn thành.
4. Phase 5 sẽ chịu trách nhiệm failure analysis, reproducibility consolidation và ablations.

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

- [x] Chọn checkpoint bằng validation metric; local demo/evaluation checkpoint epoch `20`, best valid loss `0.1547446847`.
- [x] BLEU-1..4 trên test bằng greedy và beam; report lưu ở `outputs/phase5_caption_decoding/comparison.json`.
- [ ] Perplexity hoặc test loss.
- [x] Greedy vs beam comparison trên cùng checkpoint/test split; beam tăng BLEU-4 `+0.0246`, giảm exact match `-0.0125`.
- [ ] Good/average/bad examples.
- [x] Failure records lưu repeated/missing/extra tokens và qualitative selection ở `outputs/phase5_caption_decoding/`.

### Advanced diffusion checkpoints đã thực hiện

- [ ] Sampler comparison báo cả NFE và wall-clock.
- [ ] Score/flow toy distributions có plot và numerical diagnostics.
- [ ] Pixel-vs-latent comparison tách reconstruction error khỏi generation error.
- [ ] U-Net-vs-DiT comparison ghi parameter/compute mismatch.
- [ ] One/few-step comparison báo quality/diversity degradation so với teacher.
- [x] Mọi checkpoint advanced/deferred được ghi rõ trong README và không tính vào kết quả core.

## 5.4 Checklist ablation

Chọn ít nhất 2 ablation có câu hỏi rõ ràng, không chạy chỉ để tăng số bảng.

- [ ] `32×32` vs `64×64` hoặc ghi lý do không chạy.
- [ ] Linear vs cosine beta schedule.
- [ ] U-Net không attention vs có middle attention.
- [ ] Background trắng vs neutral random.
- [x] Condition guidance scale: Phase 2/3 fixed-noise CFG boundary và scale comparisons đã có evidence; CFG `0` loại prompt/class effect đúng contract.
- [ ] Caption decoder 1 layer vs 2 layers.
- [x] Greedy vs beam search trên cùng Phase 4 checkpoint, test split và max length.

Mỗi ablation phải giữ các biến còn lại càng giống càng tốt và dùng cùng seed khi phù hợp.

## 5.5 Reproducibility checklist

- [ ] Mỗi run có config snapshot.
- [x] Seed Python/NumPy/PyTorch được cấu hình trong training/evaluation paths; demo dùng local `torch.Generator` theo seed.
- [ ] Ghi Python/PyTorch/CUDA versions.
- [ ] Ghi GPU/CPU và thời gian train.
- [ ] Checkpoint resume hoạt động.
- [ ] Raw dataset download/setup được document.
- [x] Processed dataset sinh lại bằng command + version-controlled config.
- [x] Train/evaluate/sample/demo commands được document trong README.
- [x] Không cần pretrained weights; toàn bộ core checkpoint train từ random initialization.

## 5.6 README/report checklist

- [x] Problem statement.
- [x] Scope và out-of-scope.
- [ ] Dataset và license.
- [x] Phase architecture diagram hai chiều trong README.
- [x] DDPM math notation nối với code tensor + contract table trong README.
- [ ] U-Net architecture table.
- [x] Conditioning mechanism.
- [x] Captioning architecture.
- [x] Training setup.
- [x] Metrics và protocol.
- [x] Results và ablation.
- [x] Failure cases.
- [x] Limitations.
- [x] Reproduce commands.
- [x] Future work.
- [ ] Taxonomy phân biệt formulation, representation, backbone, condition và sampler.
- [ ] Paper/code/model links trỏ tới nguồn gốc hoặc official repository/model card.
- [ ] Model lớn chỉ được dùng làm architecture study nếu không thuộc scratch experiment.

## 5.7 Definition of Done

- [x] Test suite pass trong môi trường được hỗ trợ: `208 passed` trong Conda env local.
- [x] Phase 1–4 đều có checkpoint/artifact/metric theo scope thực hiện.
- [x] Ít nhất hai ablation được báo cáo trung thực: CFG scale, greedy-vs-beam và real-vs-perturbed visual tokens.
- [ ] Có memorization/failure-case analysis.
- [x] README đủ để người khác setup, train, evaluate và chạy local playground khi có checkpoint.
- [x] Demo chỉ dùng inference adapter gọi implementation trong `src/`; không duplicate scheduler/model/attention/sampler core.

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

**Phase 4 — Image Captioning functional scratch baseline đã hoàn thành.**

Bằng chứng chính:

```text
Product captioning data
→ controlled captions reuse từ Phase 3
→ train/valid/test = 456 / 147 / 80
→ train-only vocabulary = 19 tokens
→ decoder sequence length = 10

Scratch vision path
→ image [B,3,64,64]
→ random-init CNN
→ feature map [B,256,8,8]
→ 64 spatial visual tokens [B,64,256]
→ learnable visual positional embedding

Scratch language path
→ token embedding + sinusoidal position encoding
→ custom causal self-attention
→ future-token leakage test pass
→ custom visual cross-attention
→ FFN + residual + LayerNorm
→ vocabulary logits [B,L,19]

Full end-to-end training
→ 4,386,624 parameters
→ Tesla T4
→ 100 epochs
→ best checkpoint selected only by validation loss
→ best epoch = 17
→ best valid loss = 0.1567438867
→ later epochs show clear overfitting; test uses best.pt, not epoch 100

Held-out autoregressive evaluation
→ greedy generation from BOS, no teacher forcing
→ test = 80 images
→ Exact Match = 0.3000
→ BLEU-1 = 0.8687
→ BLEU-2 = 0.7397
→ BLEU-3 = 0.5966
→ BLEU-4 = 0.4985
```

Phase 4 được đóng ở mức **functional scratch baseline with documented exceptions**.

Deferred/limitations không được claim đã hoàn thành:

- tiny overfit 1/10 samples bị người dùng chủ động bỏ qua;
- Flickr8k experiment chưa chạy;
- beam search đã implement/test và có Phase 5 greedy-vs-beam report;
- BLEU hiện là single-reference per metadata record, chưa multi-reference;
- systematic caption failure analysis đã có; bucket `average` không có sample dưới threshold hiện tại nên không claim đủ curated good/average/bad panel;
- cross-attention visualization chưa làm;
- checkpoint resume smoke chưa kiểm chứng.

Checkpoint core tiếp theo là **Phase 5 — Evaluation, Ablation, Integration and Report**.

Vertical slice tiếp theo:

```text
freeze Phase 1–4 evidence
→ lập experiment/result table thống nhất
→ captioning good/average/bad + failure analysis
→ diffusion memorization/failure summary
→ chọn ít nhất 2 ablations có câu hỏi rõ ràng và compute thấp
→ gom reproducibility commands/config/environment
→ cập nhật README/report
→ final demo chỉ gọi code trong src/
```

Không cần train lại Phase 4 trừ khi một ablation Phase 5 yêu cầu. Advanced A1–A11 vẫn là optional/deferred trừ checkpoint nào được thực sự chạy và có evidence.
