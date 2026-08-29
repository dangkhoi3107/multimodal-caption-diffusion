"""Command-line smoke check for the local demo checkpoints."""

from __future__ import annotations

import argparse

from PIL import Image

from demo.inference import get_caption_demo, get_diffusion_demo


def parse_args() -> argparse.Namespace:
    """Parse smoke-test options."""

    parser = argparse.ArgumentParser(
        description="Load both demo checkpoints and run lightweight inference."
    )
    parser.add_argument(
        "--sample-diffusion",
        action="store_true",
        help="Also run the full 1,000-step DDPM sampler (slow on CPU).",
    )
    return parser.parse_args()


def main() -> None:
    """Restore checkpoints, run captioning, and optionally sample diffusion."""

    args = parse_args()

    print("Loading captioning checkpoint...")
    caption_demo = get_caption_demo()
    print(
        "Captioning ready:",
        f"{caption_demo.parameter_count:,} parameters on {caption_demo.device}",
    )

    dummy = Image.new("RGB", (96, 64), color=(255, 255, 255))
    caption = caption_demo.generate_caption(dummy)
    print("Caption smoke output:", caption)

    print("Loading text-to-image checkpoint...")
    diffusion_demo = get_diffusion_demo()
    print(
        "Text-to-image ready:",
        f"{diffusion_demo.parameter_count:,} parameters on {diffusion_demo.device}",
    )

    if not args.sample_diffusion:
        print(
            "Skipped the full reverse chain. Pass --sample-diffusion to run "
            "all 1,000 DDPM steps."
        )
        return

    image = diffusion_demo.generate_image(
        "a red lifebuoy handwash pouch",
        guidance_scale=2.0,
        seed=42,
    )
    output_path = "demo_test_output.png"
    image.save(output_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
