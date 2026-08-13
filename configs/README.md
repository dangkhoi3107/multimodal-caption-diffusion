# Configs

Each checkpoint receives a YAML config only when that checkpoint begins. Config files describe data paths, architecture, optimization, seeds, checkpoint intervals, and output locations; they must not contain implementation logic.

Initial core files may start flat:

```text
phase0_data.yaml
phase1_unconditional.yaml
phase2_class_conditional.yaml
phase3_text_conditional.yaml
phase4_captioning.yaml
```

Khi có từ hai config trong cùng domain, chuyển sang nhóm sau thay vì tiếp tục làm root `configs/` phình to:

```text
configs/
├── data/
├── diffusion/
├── captioning/
└── experiments/
```

Advanced names và ownership được mô tả tại [repository layout](../docs/architecture/repository-layout.md). Không tạo config advanced trước khi checkpoint bắt đầu.

Every experiment must copy its resolved config into its run output directory.
