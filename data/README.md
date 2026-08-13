# Data

```text
data/
├── raw/          # Immutable downloaded/source datasets
└── processed/    # Data generated reproducibly by scripts
```

## Raw product data

The current COCO export is located at:

```text
raw/products/lifebuoy_handwash_vitamin_protection_400g/
├── coco/
│   ├── train/
│   ├── valid/
│   └── test/
└── source_export.zip
```

Do not edit files under `raw/`. Product crops, JSONL metadata, class mappings, and preprocessing summaries will be written under `processed/products_64/` by Phase 0.

Raw and processed datasets are ignored by Git. Keep download/source/license information in documentation rather than committing large image collections.
