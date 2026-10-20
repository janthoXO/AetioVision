"""RoentGen-v2 adapter (stanfordmimi/RoentGen-v2).

A complete diffusers pipeline fine-tuned from SD 2.1. The repository is gated:
accept the MIMIC-CXR terms on its Hugging Face page, then run `hf auth login`.
"""

from __future__ import annotations

import argparse
from typing import Any

from .common import Model, import_torch_diffusers, run_pipeline, select_device, select_dtype

MODEL_ID = "stanfordmimi/RoentGen-v2"


def generate(prompt: str, args: argparse.Namespace) -> Any:
    torch, diffusers = import_torch_diffusers()
    device = select_device(torch, args.device)
    dtype = select_dtype(torch, args.dtype, device)
    print(f"Loading {MODEL_ID} on {device} with {dtype}...")

    try:
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=dtype, use_safetensors=True, safety_checker=None
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not load {MODEL_ID}. It is gated: request access at "
            f"https://huggingface.co/{MODEL_ID} (requires MIMIC-CXR credentials), "
            "then run 'hf auth login'."
        ) from exc
    return run_pipeline(torch, pipe, prompt, args, device)


MODELS = [
    Model(
        key="roentgen",
        name="RoentGen-v2",
        body_parts="Chest / thorax only",
        modalities="Chest X-ray (frontal radiograph)",
        generate=generate,
    )
]
