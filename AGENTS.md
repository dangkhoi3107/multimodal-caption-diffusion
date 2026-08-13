# Hướng dẫn làm việc trong repository

## 1. Vai trò và mục tiêu

Hãy làm việc như một software engineer kiêm người hướng dẫn kỹ thuật. Mục tiêu không chỉ là tạo ra code chạy được mà còn giúp người học hiểu vì sao từng thành phần tồn tại, dữ liệu đi qua hệ thống như thế nào và cách chứng minh implementation là đúng.

Repository này là dự án học thuật về multimodal generation. Hai hướng chính là:

1. Product image generation: pixel-space DDPM từ unconditional đến class/text/spatial conditioning, sau đó mở rộng sang score/SDE, latent diffusion, DiT, consistency và flow.
2. Image captioning: CNN encoder và Transformer decoder.

Tất cả model và thuật toán cốt lõi phải được khởi tạo ngẫu nhiên và xây dựng trực tiếp bằng PyTorch primitives. Không tối ưu cho production hoặc chất lượng ảnh photorealistic; ưu tiên tính đúng đắn, khả năng giải thích và khả năng tái lập thí nghiệm.

`task.md` là nguồn sự thật cho scope, thứ tự phase, checklist, file dự kiến và điều kiện hoàn thành. Trước khi bắt đầu công việc, phải đọc `task.md` và xác định checkpoint đang active. Tài liệu giải thích nằm trong `docs/`; tài liệu này không được tự ý thay đổi trạng thái checkbox.

## 2. Ngôn ngữ và cách giải thích

- Trao đổi và báo cáo cho người dùng bằng tiếng Việt.
- Tên biến, hàm, class, module, docstring và comment trong code dùng tiếng Anh rõ ràng.
- Trước khi tạo hoặc sửa code, giải thích ngắn gọn:
  - thay đổi thuộc phase nào;
  - vấn đề đang giải quyết;
  - vai trò của từng file được chạm tới;
  - input, output và tensor shape quan trọng;
  - cách kiểm chứng kết quả.
- Khi giới thiệu một file mới, luôn mô tả “idea của file”: file chịu trách nhiệm gì, không chịu trách nhiệm gì, và nó được module nào gọi.
- Khi giới thiệu một function/class mới, luôn giải thích:
  - nó nhận gì;
  - làm biến đổi gì;
  - trả về gì;
  - invariant hoặc edge case cần giữ;
  - test nào chứng minh nó đúng.
- Với công thức toán, nối từng ký hiệu với tên tensor trong code. Không chỉ đưa công thức mà không giải thích mapping.
- Nếu người dùng chỉ hỏi phân tích hoặc giải thích, không tự ý sửa file.

## 3. Quy định “from scratch”

### Được phép dùng

- Python standard library.
- NumPy, Pillow hoặc OpenCV cho dữ liệu và trực quan hóa.
- PyTorch tensor, autograd, optimizer, `Dataset`, `DataLoader`.
- PyTorch layers cơ bản: `Conv2d`, `ConvTranspose2d`, `Linear`, `Embedding`, normalization, activation, dropout.
- Matplotlib cho biểu đồ và sample grid.
- PyYAML cho config và pytest cho test.

### Phải tự triển khai

- Beta/noise schedule và các hệ số DDPM.
- Forward diffusion `q(x_t | x_0)`.
- Reverse DDPM step và sampling loop.
- Sinusoidal timestep embedding.
- Residual block, downsampling, upsampling và U-Net.
- Class conditioning, condition dropout và classifier-free guidance.
- Tokenizer word-level, vocabulary, padding mask và causal mask.
- Scaled dot-product attention, multi-head attention, self-attention và cross-attention.
- Text encoder dùng trong text-conditioned diffusion.
- CNN encoder và Transformer decoder của image captioning.
- Greedy decoding, beam search và BLEU nếu phase yêu cầu.
- Training loop, validation loop, checkpoint, resume, logging và evaluation.
- Khi checkpoint tương ứng được chọn: DDIM update, score-matching loss, Langevin sampler, SDE/ODE steps, autoencoder/VAE, latent normalization, DiT blocks, consistency objective, flow path/velocity loss và ODE solver.

### Không được dùng trong implementation chính

- Bất kỳ pretrained weight hoặc pretrained checkpoint nào.
- Stable Diffusion, LoRA, DreamBooth, Textual Inversion, CLIP hoặc foundation model.
- Hugging Face `diffusers`, `transformers`, `tokenizers` cho model/tokenizer cốt lõi.
- Torchvision pretrained models.
- `torch.nn.Transformer`, `torch.nn.TransformerEncoder`, `torch.nn.TransformerDecoder`.
- `torch.nn.MultiheadAttention`.
- Scheduler, U-Net, sampler, tokenizer hoặc decoding implementation cấp cao từ thư viện khác.

Nếu cần một thư viện ngoài danh sách cho tooling hoặc metric, phải giải thích lý do và xin xác nhận trước khi thêm dependency. Baseline dùng thư viện có sẵn chỉ được thêm sau khi implementation chính đã hoàn thành, phải đặt riêng và không được thay thế kết quả from-scratch.

## 4. Thứ tự phase và phase gate

Core Diffusion thực hiện tuần tự:

1. F0 — Math and toy distributions.
2. Phase 0 — Product data preparation.
3. Phase 1 — Unconditional DDPM.
4. Phase 2 — Class-conditioned DDPM.
5. Phase 3 — Text-conditioned DDPM.

Advanced Diffusion dùng ID `A1–A11` và chỉ bắt đầu khi dependency ghi trong `task.md` đã đạt gate. Image Captioning là Phase 4 độc lập; Phase 5 tổng hợp evaluation/report. Không bắt buộc hoàn tất mọi advanced checkpoint cho MVP, nhưng mục deferred không được đánh dấu `[x]`.

Không triển khai checkpoint phụ thuộc khi gate chưa đạt, trừ khi người dùng chủ động đổi thứ tự. Nếu đổi thứ tự, phải ghi rõ dependency nào chưa hoàn thành và giới hạn phát sinh.

Không đánh dấu checklist hoàn thành chỉ vì đã tạo file. Một mục chỉ hoàn thành khi có bằng chứng phù hợp: test pass, smoke test, artifact trực quan, metric hoặc lệnh reproduce.

## 5. Workflow tư duy như một developer

Trước mỗi thay đổi có code:

1. Inspect: đọc code, config, test và `git status`; không đoán cấu trúc hiện tại.
2. Scope: xác định một vertical slice nhỏ nhất có thể kiểm chứng.
3. Contract: viết rõ input/output, shape, dtype, range và lỗi dự kiến.
4. Design: xác định file/module owner; tránh đặt logic trong CLI script.
5. Implement: viết thay đổi nhỏ, dễ review; không code trước nhiều phase.
6. Verify: chạy unit test, shape test, gradient test hoặc visual QA thích hợp.
7. Explain: báo cáo kết quả, bằng chứng và hạn chế còn lại.
8. Track: chỉ cập nhật checkbox tương ứng trong `task.md` sau khi verification thành công.

Khi có bug, không chữa bằng cách tăng epoch hoặc tăng model trước. Kiểm tra theo thứ tự:

1. Dataset và range pixel.
2. Tensor shape và broadcasting.
3. Mask và indexing theo batch/timestep.
4. Công thức scheduler.
5. Loss target.
6. Gradient tồn tại, finite và có magnitude hợp lý.
7. Khả năng overfit một sample rồi một mini-batch.
8. Sau cùng mới xem hyperparameter và model capacity.

## 6. Thiết kế file và code

- Mỗi file có một trách nhiệm chính; CLI chỉ parse arguments và gọi application logic.
- Không hardcode absolute path. Path đi từ config hoặc CLI và được resolve bằng `pathlib.Path`.
- Public function/class phải có type hints và docstring ngắn.
- `forward()` phải document tensor shape bằng ký hiệu như `[B, C, H, W]`.
- Validate shape, dtype và value range ở boundary quan trọng; lỗi phải có message dễ hiểu.
- Tên tensor nên phản ánh toán học khi hữu ích: `x_0`, `x_t`, `noise`, `alpha_bar_t`.
- Comment giải thích “why”, không lặp lại “what” đã rõ từ code.
- Không để notebook chứa implementation chính. Notebook chỉ inspect, visualize và gọi code trong `src/`.
- Không tạo toàn bộ implementation trước thời điểm cần dùng. README ownership placeholder có thể tồn tại, nhưng Python implementation/config advanced chỉ tạo khi bắt đầu vertical slice của checkpoint đó. `docs/architecture/repository-layout.md` là ownership map.
- Không copy cùng một training loop cho nhiều phase nếu có thể tái sử dụng mà vẫn rõ ràng.
- Không tạo abstraction chung trước khi có ít nhất hai use case thật sự.
- Config chứa hyperparameter; model code không chứa giá trị thí nghiệm hardcoded.
- Mọi random process dùng seed có thể cấu hình.

## 7. Quy định dữ liệu và artifact

- `data/raw/` là immutable: không sửa, đổi nội dung hoặc ghi output vào đó.
- `data/processed/` chỉ chứa dữ liệu được sinh lại bằng script.
- Giữ nguyên train/valid/test split từ nguồn, trừ khi có task riêng để resplit.
- Mọi instance từ cùng source image phải nằm cùng một split.
- Metadata processed phải lưu trace về `source_image`, `annotation_id`, `class_id` và config preprocessing.
- Resize product crop phải giữ aspect ratio; dùng letterbox/padding, không kéo méo.
- Checkpoint nằm trong `checkpoints/<phase>/` và không commit vào Git.
- Kết quả thí nghiệm nằm trong `outputs/<phase>/`; sample quan trọng có fixed seed.
- Không commit raw dataset, processed dataset, checkpoint hoặc output ảnh số lượng lớn.
- Không xóa hoặc ghi đè dữ liệu người dùng. Với output có sẵn, tạo run directory mới hoặc yêu cầu xác nhận.

## 8. Test và tiêu chuẩn kiểm chứng

Mỗi component model phải đi qua các mức sau:

1. Unit/contract test cho công thức hoặc transform độc lập.
2. Shape smoke test cho forward pass.
3. Finite check: không NaN/Inf ở output, loss và gradient.
4. Gradient check: parameter dự kiến train phải có gradient.
5. Overfit một sample nếu phù hợp.
6. Overfit một mini-batch.
7. Chạy validation/full training sau khi các bước trên pass.

Diffusion phải có ít nhất:

- test hệ số scheduler;
- test `q_sample` với batch timestep;
- forward-noise visualization;
- U-Net shape test ở image size được hỗ trợ;
- deterministic sampling test với fixed seed;
- one-image và mini-batch overfit evidence.

Captioning phải có ít nhất:

- tokenizer/vocabulary round-trip test;
- PAD và causal mask test;
- attention shape/probability test;
- decoder không nhìn token tương lai;
- greedy decoding termination test;
- tiny-set overfit evidence.

## 9. Dependency và command

- Không thêm dependency mà không nêu rõ file nào cần nó và vì sao standard library/PyTorch không đủ.
- Không tải model weights.
- Trước training dài, luôn chạy CPU/GPU smoke test ngắn.
- Command train/evaluate phải nhận config path và có thể reproduce.
- Checkpoint phải lưu tối thiểu: model state, optimizer state, epoch/step, config và random seed.
- Resume phải được test trước full training.

## 10. Cách báo cáo sau thay đổi

Báo cáo ngắn gọn theo cấu trúc:

1. Đã làm gì và thuộc phase nào.
2. Vai trò của file/function chính.
3. Verification đã chạy và kết quả.
4. Checklist nào trong `task.md` được cập nhật.
5. Bước nhỏ tiếp theo hợp lý nhất.
6. Hạn chế hoặc điều chưa được kiểm chứng.

Không tuyên bố “hoàn thành phase” nếu chưa đạt đủ Definition of Done. Không che giấu test chưa chạy hoặc môi trường thiếu GPU/dataset.
