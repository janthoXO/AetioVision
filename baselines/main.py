#!/usr/bin/env python3
"""Pick a 2D medical text-to-image model, write a prompt, and save the result.

Each model adapter lives in its own module under ``models/``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from models import MODELS
from models.common import Model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a timestamped synthetic medical image with a chosen model."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text describing the image. If omitted, read one prompt interactively.",
    )
    parser.add_argument(
        "--model",
        choices=[m.key for m in MODELS],
        help="Model key. If omitted, choose from an interactive menu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated images (default: outputs).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Execution device (default: auto; prefers MPS on Apple Silicon).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model precision (default: auto; float32 on MPS/CPU).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion steps (default: 30).",
    )
    parser.add_argument("--seed", type=int, help="Optional reproducibility seed.")
    parser.add_argument(
        "--no-attention-slicing",
        action="store_true",
        help="Disable attention slicing; uses more memory but may be faster.",
    )
    return parser


def choose_model() -> Model:
    print("Available models:")
    for i, m in enumerate(MODELS, 1):
        print(f"  {i:>2}) {m.name}\n      body parts: {m.body_parts}\n      modality:   {m.modalities}")
    while True:
        choice = input(f"Select model [1-{len(MODELS)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            return MODELS[int(choice) - 1]
        print("Invalid choice.")


def timestamped_path(output_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"{prefix}_{timestamp}.png"


def main() -> int:
    args = build_parser().parse_args()
    if args.steps < 1:
        print("error: --steps must be at least 1", file=sys.stderr)
        return 2

    model = next((m for m in MODELS if m.key == args.model), None) or choose_model()

    if not args.prompt:
        print(f"\n{model.name} covers: {model.body_parts} ({model.modalities})")
        args.prompt = input("Prompt: ").strip()
    if not args.prompt:
        print("error: prompt cannot be empty", file=sys.stderr)
        return 2

    try:
        image = model.generate(args.prompt, args)
        output_path = timestamped_path(args.output_dir, model.key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
