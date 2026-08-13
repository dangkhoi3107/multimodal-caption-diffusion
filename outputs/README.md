# Outputs

Experiment artifacts are grouped by checkpoint and run ID. Existing Phase 0–5 folders remain valid; advanced folders are created only when an experiment starts:

```text
outputs/
├── phase0_data/<run_id>/
├── phase1_unconditional/<run_id>/
├── phase2_class_conditional/<run_id>/
├── phase3_text_conditional/<run_id>/
├── phase4_captioning/<run_id>/
├── phase5_evaluation/<run_id>/
├── f0_math_toy/<run_id>/
├── a1_improved_design/<run_id>/
├── a2_fast_sampling/<run_id>/
└── a3...a11/<run_id>/
```

Each run should include `config.resolved.yaml`, `environment.json`, `metrics.jsonl`, `command.txt`, visual evidence/failure cases and `summary.md`. Checkpoints belong under `checkpoints/`, not here. Exact artifact contracts are in [repository layout](../docs/architecture/repository-layout.md).
