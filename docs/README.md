# Documentation map

Thư mục này chứa phần giải thích dài hạn của project. `task.md` ở repository root vẫn là nguồn sự thật duy nhất cho trạng thái checkbox và phase gate.

## Đọc theo thứ tự

1. [Diffusion curriculum](roadmap/diffusion-curriculum.md): lộ trình từ Gaussian toy đến DDPM, LDM, DiT, Consistency và Rectified Flow.
2. [Model and paper catalog](references/model-catalog.md): phân loại đúng, paper gốc, official code và model card.
3. [Repository layout](architecture/repository-layout.md): module owner, file dự kiến, input/output và dependency giữa các package.
4. [Captioning curriculum](roadmap/captioning-curriculum.md): track Image → Text độc lập.
5. [Root task checklist](../task.md): checklist đang thực thi và Definition of Done.

## Trách nhiệm của từng tài liệu

- `task.md`: việc gì đã/chưa hoàn thành, bằng chứng và gate.
- `docs/roadmap/`: vì sao học theo thứ tự đó và checkpoint chứng minh điều gì.
- `docs/references/`: link đọc/tham khảo; không phải dependency của implementation.
- `docs/architecture/`: file nào sở hữu logic nào; không chứa trạng thái hoàn thành.
- `README.md`: giới thiệu ngắn và điểm bắt đầu cho người mới.

Không đánh dấu một checkpoint hoàn thành chỉ vì đã đọc paper hoặc tạo skeleton file.
