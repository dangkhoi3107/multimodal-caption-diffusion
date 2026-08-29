from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.captioning.generation import (
    greedy_generate,
)
from src.captioning.metrics import (
    corpus_bleu,
)
from src.captioning.model import CaptionModel
from src.data.caption_dataset import CaptionDataset
from src.text.tokenizer import (
    decode,
    tokenize,
)
from src.text.vocabulary import Vocabulary


CONFIG_PATH = Path(
    "configs/phase4_captioning.yaml"
)

CHECKPOINT_PATH = Path(
    "outputs/phase4_captioning/best.pt"
)

OUTPUT_ROOT = Path(
    "outputs/phase4_captioning_evaluation"
)


def restore_vocabulary(
    data: dict,
) -> Vocabulary:
    vocabulary = Vocabulary(
        token_to_id={
            str(token): int(token_id)
            for token, token_id
            in data["token_to_id"].items()
        },
        id_to_token=tuple(
            str(token)
            for token
            in data["id_to_token"]
        ),
    )

    vocabulary.validate()

    return vocabulary


def build_model(
    config: dict,
    vocabulary: Vocabulary,
) -> CaptionModel:
    return CaptionModel(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        image_size=int(
            config["data"]["image_size"]
        ),
        in_channels=int(
            config["model"]["image_encoder"]["in_channels"]
        ),
        base_channels=int(
            config["model"]["image_encoder"]["base_channels"]
        ),
        model_dim=int(
            config["model"]["model_dim"]
        ),
        max_length=int(
            config["text"]["sequence_length"]
        ),
        num_heads=int(
            config["model"]["decoder"]["num_heads"]
        ),
        num_layers=int(
            config["model"]["decoder"]["num_layers"]
        ),
        feedforward_dim=int(
            config["model"]["decoder"]["feedforward_dim"]
        ),
        dropout=float(
            config["model"]["decoder"]["dropout"]
        ),
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    vocabulary = restore_vocabulary(
        checkpoint["vocabulary"]
    )

    dataset = CaptionDataset(
        metadata_path=Path(
            config["data"]["test_metadata"]
        ),
        vocabulary=vocabulary,
        sequence_length=int(
            config["text"]["sequence_length"]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            config["training"]["batch_size"]
        ),
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        config,
        vocabulary,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    references = []
    hypotheses = []
    rows = []

    exact_matches = 0
    total = 0

    generation_max_length = int(
        config["generation"]["max_length"]
    )

    for batch in loader:
        images = batch["image"].to(device)

        generated = greedy_generate(
            model=model,
            images=images,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
            pad_id=vocabulary.pad_id,
            max_length=generation_max_length,
        )

        generated = generated.detach().cpu()

        batch_size = images.shape[0]

        for index in range(batch_size):
            reference = str(
                batch["caption"][index]
            )

            hypothesis = decode(
                generated[index],
                vocabulary=vocabulary,
            )

            reference_tokens = tokenize(
                reference
            )

            hypothesis_tokens = tokenize(
                hypothesis
            )

            references.append(
                reference_tokens
            )

            hypotheses.append(
                hypothesis_tokens
            )

            exact = (
                reference_tokens
                == hypothesis_tokens
            )

            exact_matches += int(exact)
            total += 1

            rows.append(
                {
                    "file_name": str(
                        batch["file_name"][index]
                    ),
                    "class_id": int(
                        batch["class_id"][index].item()
                    ),
                    "reference": reference,
                    "prediction": hypothesis,
                    "exact_match": exact,
                }
            )

    bleu = corpus_bleu(
        references=references,
        hypotheses=hypotheses,
        max_n=4,
    )

    exact_match_accuracy = (
        exact_matches / total
    )

    report = {
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": checkpoint[
            "epoch"
        ],
        "test_samples": total,
        "exact_match_accuracy": (
            exact_match_accuracy
        ),
        **bleu,
        "predictions": rows,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_ROOT
        / "report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )
    print(
        "Test samples:",
        total,
    )
    print(
        "Exact match:",
        f"{exact_match_accuracy:.4f}",
    )
    print(
        "BLEU-1:",
        f"{bleu['bleu1']:.4f}",
    )
    print(
        "BLEU-2:",
        f"{bleu['bleu2']:.4f}",
    )
    print(
        "BLEU-3:",
        f"{bleu['bleu3']:.4f}",
    )
    print(
        "BLEU-4:",
        f"{bleu['bleu4']:.4f}",
    )

    print()
    print("Examples:")

    for row in rows[:12]:
        print(
            f"[class={row['class_id']}] "
            f"REF: {row['reference']}"
        )
        print(
            f"          PRED: "
            f"{row['prediction']}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_ROOT / "report.json",
    )


if __name__ == "__main__":
    main()
