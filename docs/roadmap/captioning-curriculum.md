# Image captioning curriculum

Track này độc lập với diffusion, nhưng tái sử dụng tokenizer, attention primitives, config, checkpoint và evaluation utilities của repository.

## Dependency

```text
caption dataset
→ vocabulary/tokenizer
→ CNN spatial encoder
→ causal Transformer decoder
→ greedy decoding
→ beam search
→ BLEU + qualitative evaluation
```

## C0 — Data và text contracts

- Chọn dataset có license và split rõ.
- Build vocabulary chỉ từ train captions.
- Encode `<BOS> ... <EOS>`, pad theo batch.
- Test round-trip, unknown token, truncation và padding mask.

## C1 — CNN visual encoder

- CNN random initialization, không pretrained backbone.
- Giữ spatial feature map rồi flatten thành visual tokens `[B, N, D]`.
- Thêm/project positional information.
- Test gradient từ caption loss đi về convolution đầu tiên.

## C2 — Transformer decoder

- Tự viết scaled dot-product attention và multi-head reshape/merge.
- Masked self-attention không được nhìn token tương lai.
- Cross-attention dùng text states làm query và visual tokens làm key/value.
- Vocabulary head trả logits `[B, L, V]`.

## C3 — Training và decoding

- Teacher-forcing input là caption shift-right.
- Cross-entropy bỏ qua PAD.
- Overfit một image-caption pair rồi tiny dataset.
- Greedy decode phải dừng ở EOS.
- Beam search chỉ thêm sau greedy tests.

## C4 — Evaluation

- BLEU-1..4 với nhiều reference captions.
- Test loss/perplexity.
- Good/average/bad examples.
- Phân tích repetition, hallucination và generic captions.
- Attention map chỉ là diagnostic, không tự động được diễn giải như explanation nhân quả.

Checklist chính thức của track nằm trong [task.md](../../task.md).
