"""MediSyn adapter (hiesingerlab/MediSyn).

MediSyn publishes only its fine-tuned SD 1.x UNet. The base pipeline supplies
the tokenizer, CLIP text encoder, scheduler, and VAE.
"""

from __future__ import annotations

import argparse
from typing import Any

from .common import Model, import_torch_diffusers, run_pipeline, select_device, select_dtype

MODEL_ID = "hiesingerlab/MediSyn"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def generate(prompt: str, args: argparse.Namespace) -> Any:
    torch, diffusers = import_torch_diffusers()
    device = select_device(torch, args.device)
    dtype = select_dtype(torch, args.dtype, device)
    print(f"Loading MediSyn UNet plus {BASE_MODEL} on {device} with {dtype}...")

    unet = diffusers.UNet2DConditionModel.from_pretrained(
        MODEL_ID, torch_dtype=dtype, use_safetensors=True
    )
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, unet=unet, torch_dtype=dtype, use_safetensors=True, safety_checker=None
    )
    return run_pipeline(torch, pipe, prompt, args, device)


MODELS = [
    Model(
        key="medisyn",
        name="MediSyn",
        body_parts="Generalist: radiology, pathology, dermatology, ophthalmology, "
        "gastroenterology, surgery",
        modalities="CT, X-ray, MRI, ultrasound, endoscopy, microscopy, fundus, OCT, "
        "dermoscopy, clinical photos",
        generate=generate,
    )
]
