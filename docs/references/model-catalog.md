# Diffusion model, paper and official implementation catalog

Catalog này đầy đủ theo scope học tập của repository, không phải danh sách mọi diffusion paper đã xuất bản. Link code/model chỉ dùng để đối chiếu sau khi implementation from-scratch tương ứng đã có test; không copy implementation hoặc tải pretrained weights vào experiment chính.

## 1. Cách phân loại đúng

Một hệ thống sinh ảnh có thể đồng thời thuộc nhiều cột dưới đây:

| Trục | Câu hỏi | Ví dụ |
|---|---|---|
| Formulation | Quá trình xác suất/ODE được định nghĩa thế nào? | DDPM, score SDE, consistency, flow matching |
| Representation | Quá trình sinh chạy ở đâu? | pixel space, latent space |
| Backbone | Mạng dự đoán noise/score/velocity là gì? | U-Net, DiT, MMDiT |
| Conditioning | Model được điều khiển bằng gì? | none, class, text, mask, edge, depth |
| Parameterization | Network dự đoán đại lượng nào? | epsilon, x0, v, score, velocity field |
| Sampler/solver | Inference giải reverse process bằng cách nào? | ancestral DDPM, DDIM, PNDM, DPM-Solver, Euler/ODE solver |
| Training/fine-tuning method | Trọng số được học hoặc thích nghi thế nào? | full training, CFG, distillation, LoRA, Textual Inversion |
| Product/checkpoint | Tên model cụ thể nào? | Stable Diffusion, SD3, FLUX.1 |

Vì vậy `LDM`, `DiT`, `DDIM` và `ControlNet` không phải bốn thế hệ kế tiếp của cùng một loại: chúng lần lượt nói về representation, backbone, sampler và spatial conditioning.

## 2. Nền tảng xác suất và DDPM

### Diffusion Probabilistic Models — 2015

- Loại: formulation lịch sử.
- Học gì: forward diffusion phá cấu trúc dữ liệu và learned reverse process khôi phục dữ liệu.
- Paper: [Deep Unsupervised Learning using Nonequilibrium Thermodynamics — ICML 2015](https://proceedings.mlr.press/v37/sohl-dickstein15.html).
- Ghi chú: công trình ở ICML 2015, không phải NIPS 2015.

### DDPM — 2020

- Loại: discrete-time diffusion formulation + ancestral sampler.
- Học gì: beta schedule, closed-form `q(x_t|x_0)`, epsilon prediction và reverse Markov chain.
- Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Project page: [hojonathanho.github.io/diffusion](https://hojonathanho.github.io/diffusion/).
- Official code: [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion).
- Online checkpoint để xem kiến trúc/model card, không dùng trong scratch run: [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32).

### Improved DDPM — 2021

- Loại: training/objective/schedule improvements.
- Học gì: cosine schedule, learned reverse variance, hybrid objective và timestep importance sampling.
- Paper: [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a.html).
- Official code: [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

### EDM — 2022

- Loại: design-space/preconditioning/training and sampling formulation.
- Học gì: tách rõ data/noise scaling, preconditioning, noise-level sampling và sampler design.
- Paper: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).
- Official code: [NVlabs/edm](https://github.com/NVlabs/edm).
- Vai trò trong project: checkpoint nghiên cứu sau khi DDPM baseline đã đúng; không thay DDPM core.

## 3. Sampler và solver nhanh

### DDIM — 2020/ICLR 2021

- Loại: implicit/non-Markovian sampling process.
- Học gì: dùng model train bằng DDPM objective nhưng sampling với ít bước hơn; `eta=0` cho đường deterministic.
- Paper: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).
- Official code: [ermongroup/ddim](https://github.com/ermongroup/ddim).

### PNDM — ICLR 2022

- Loại: pseudo numerical sampler.
- Paper: [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
- Official code: [luping-liu/PNDM](https://github.com/luping-liu/PNDM).
- Vai trò: optional sau DDIM; không cần để chứng minh DDPM hoạt động.

### DPM-Solver — 2022

- Loại: high-order ODE solver chuyên cho diffusion ODE.
- Paper: [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927).
- Official code: [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver).
- Vai trò: optional advanced sampler; chỉ implement sau khi DDIM có test deterministic.

## 4. Conditioning và text-to-image

### Classifier-Free Guidance — 2021/2022

- Loại: conditioning/training and sampling technique.
- Học gì: jointly train conditional/unconditional predictions bằng condition dropout rồi kết hợp hai prediction khi inference.
- Paper: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).

### GLIDE — 2021

- Loại: text-conditioned pixel diffusion product/research model.
- Paper: [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741).
- Official code: [openai/glide-text2im](https://github.com/openai/glide-text2im).
- Trạng thái link: official repository đã được archive/read-only ngày 29/05/2026 nhưng vẫn là nguồn tham khảo gốc.
- Vai trò: kiến trúc tham khảo cho text conditioning/CFG; không phải checkpoint cần tái tạo ở scale gốc.

### Imagen — 2022

- Loại: cascaded text-to-image diffusion system.
- Paper/project page: [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://imagen.research.google/).
- Paper PDF: [arXiv:2205.11487](https://arxiv.org/abs/2205.11487).
- Ghi chú: dùng pretrained T5 text encoder; không có official open training checkpoint tương đương và không phù hợp rule scratch của project.

### ControlNet — ICCV 2023

- Loại: spatial conditioning adapter, không phải diffusion formulation mới.
- Paper: [Adding Conditional Control to Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html).
- Official code: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet).
- Áp dụng cho project: dùng polygon mask/edge làm control map sau khi base text-conditioned model của chính project đã hoạt động.

## 5. Score-based models và continuous time

### NCSN — NeurIPS 2019

- Loại: score-based generative formulation.
- Học gì: estimate `∇x log p_sigma(x)` ở nhiều noise scale và sampling bằng annealed Langevin dynamics.
- Paper: [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600).
- Official code: [ermongroup/ncsn](https://github.com/ermongroup/ncsn).

### Score SDE — ICLR 2021

- Loại: continuous-time framework thống nhất score model và diffusion.
- Học gì: forward SDE, reverse-time SDE, predictor-corrector và probability-flow ODE.
- Paper: [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS).
- Official JAX code: [yang-song/score_sde](https://github.com/yang-song/score_sde).
- Official PyTorch code: [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).

## 6. Latent representation

### VAE foundation — 2013/2014

- Loại: learned latent representation.
- Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Vai trò: hiểu reconstruction, posterior, KL và reparameterization trước latent diffusion.

### Latent Diffusion Models — CVPR 2022

- Loại: diffusion trong learned spatial latent tensor.
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html).
- Official code: [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).
- Stable Diffusion reference code: [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
- Online model card tham khảo, không dùng trong scratch run: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4).
- Ghi chú: phải train và kiểm tra autoencoder trước; diffusion không thể phục hồi chi tiết mà decoder đã làm mất.

## 7. Transformer backbone

### DiT — 2022/ICCV 2023

- Loại: backbone; thay U-Net bằng Transformer trên latent patches.
- Paper: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
- Official code/weights: [facebookresearch/DiT](https://github.com/facebookresearch/DiT).
- Vai trò: so sánh backbone dưới cùng diffusion objective, data và budget; không phải formulation mới.

## 8. One-step/few-step generation

### Consistency Models — ICML 2023

- Loại: family/training objective cho one-step hoặc few-step generation.
- Paper: [Consistency Models](https://proceedings.mlr.press/v202/song23a.html).
- Official code: [openai/consistency_models](https://github.com/openai/consistency_models).
- Online model card tham khảo: [openai/diffusers-cd_imagenet64_l2](https://huggingface.co/openai/diffusers-cd_imagenet64_l2).
- Hướng học: consistency distillation từ teacher của project trước; standalone consistency training là extension.

## 9. Flow Matching và Rectified Flow

### Flow Matching — ICLR 2023

- Loại: continuous normalizing flow training framework.
- Paper: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Official guide/code: [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).
- Học gì: probability path, conditional velocity target, vector-field regression và ODE sampling.

### Rectified Flow — ICLR 2023

- Loại: flow/transport formulation với mục tiêu trajectory thẳng hơn.
- Paper: [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003).
- Official code: [gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow).
- Học gì: linear interpolation target, Euler sampling, reflow và distillation tùy chọn.

### Stable Diffusion 3 — 2024

- Loại: sản phẩm/architecture study; latent rectified flow + MMDiT + nhiều text encoder.
- Technical paper: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).
- Official model card: [stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium).
- Vai trò: đọc để hiểu cách ghép latent representation, flow objective và multimodal Transformer; không tái tạo scale gốc.

### FLUX.1 — 2024

- Loại: model/product cụ thể dựa trên latent rectified-flow Transformer.
- Official inference code: [black-forest-labs/flux](https://github.com/black-forest-labs/flux).
- Open model card: [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- Vai trò: architecture/model-card study. Model 12B không phải target train-from-scratch của repository.

## 10. Fine-tuning methods nằm ngoài implementation chính

Các phương pháp này giả định đã có một base model mạnh. Chúng không thay thế curriculum train-from-scratch:

| Phương pháp | Nó học gì? | Nguồn |
|---|---|---|
| Textual Inversion | Một/vài embedding token mới, giữ phần lớn base model cố định | [Paper](https://arxiv.org/abs/2208.01618) |
| DreamBooth | Fine-tune text-to-image model để gắn identifier hiếm với subject | [Paper](https://arxiv.org/abs/2208.12242) |
| LoRA | Low-rank weight updates cho parameter-efficient fine-tuning | [Paper](https://arxiv.org/abs/2106.09685) |

Token `<prod_handwash>` trong model tự train của project chỉ là token vocabulary bình thường. Nó chỉ trở thành “learned special concept token” theo nghĩa Textual Inversion khi base model bị freeze và riêng embedding token được tối ưu theo objective đó.

## 11. Thứ tự đọc tối thiểu

1. DDPM.
2. Improved DDPM.
3. DDIM.
4. Classifier-Free Guidance.
5. NCSN và Score SDE.
6. LDM.
7. DiT.
8. Consistency Models.
9. Flow Matching và Rectified Flow.
10. SD3/FLUX model cards để thấy các trục được ghép thành hệ thống lớn như thế nào.
