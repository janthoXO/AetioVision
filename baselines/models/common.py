"""Shared pieces for the model adapters: the menu entry type and M3-friendly
device, dtype, and pipeline helpers.

PyTorch and Diffusers are imported lazily inside functions so that
``python main.py --help`` works before the virtual environment is set up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    body_parts: str
    modalities: str
    # (prompt, CLI args) -> PIL.Image.Image
    generate: Callable[[str, argparse.Namespace], Any]


def import_torch_diffusers() -> tuple[Any, Any]:
    try:
        import diffusers
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Activate the virtual environment and run "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    return torch, diffusers


def select_device(torch: Any, requested: str) -> Any:
    """Return a usable torch.device, with an M3-friendly automatic default."""
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    cuda_available = bool(torch.cuda.is_available())

    if requested == "auto":
        if mps_available:
            return torch.device("mps")
        if cuda_available:
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but this PyTorch build cannot use MPS.")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def select_dtype(torch: Any, requested: str, device: Any) -> Any:
    """Choose precision that is supported reliably by Apple Silicon."""
    if requested != "auto":
        return getattr(torch, requested)
    if device.type in {"mps", "cpu"}:
        # All adapters ship F32 weights; float16/bfloat16 on MPS often yields
        # black images or NaNs with SD-based pipelines.
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else torch.float32


def run_pipeline(torch: Any, pipe: Any, prompt: str, args: argparse.Namespace, device: Any) -> Any:
    """Move a Stable Diffusion pipeline to the device and generate one image."""
    pipe = pipe.to(device)
    if device.type == "mps" and not args.no_attention_slicing:
        pipe.enable_attention_slicing()

    generator = None
    if args.seed is not None:
        # CPU noise transfers reliably to MPS/CUDA and keeps seeds reproducible.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(prompt, num_inference_steps=args.steps, generator=generator)
    return result.images[0]
