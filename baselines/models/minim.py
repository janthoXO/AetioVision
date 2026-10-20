"""MINIM adapter (https://github.com/WithStomach/MINIM).

MINIM fine-tunes SD v1-4 UNets. The weights are only shared via Baidu Pan, in
the RunMINIM bundle next to run_minim.exe, as four UNets under ``model/unets``.
The maintainer's mapping (GitHub issue #8) is 1: OCT, 2: CXR, 3: fundus,
4: MRI + CT. Copy each UNet folder (config.json + diffusion_pytorch_model.*) to

    weights/minim/{oct, chest-xray, fundus, mri-ct}/
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

from .common import Model, import_torch_diffusers, run_pipeline, select_device, select_dtype

BASE_MODEL = "CompVis/stable-diffusion-v1-4"
WEIGHTS_DIR = Path("weights/minim")


def generate(weights: str, prompt: str, args: argparse.Namespace) -> Any:
    unet_dir = WEIGHTS_DIR / weights
    if not (unet_dir / "config.json").is_file():
        raise RuntimeError(
            f"MINIM weights not found at {unet_dir}. Download RunMINIM from "
            "https://pan.baidu.com/s/1cxeonEof1oJRMF_Y-rF0UA?pwd=1234 and copy the "
            f"'{weights}' UNet from model/unets there (1: oct, 2: chest-xray, "
            "3: fundus, 4: mri-ct)."
        )

    torch, diffusers = import_torch_diffusers()
    device = select_device(torch, args.device)
    dtype = select_dtype(torch, args.dtype, device)
    print(f"Loading MINIM {weights} UNet plus {BASE_MODEL} on {device} with {dtype}...")

    # No use_safetensors: MINIM checkpoints may be .bin, diffusers picks what exists.
    unet = diffusers.UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype)
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, unet=unet, torch_dtype=dtype, safety_checker=None
    )
    return run_pipeline(torch, pipe, prompt, args, device)


# (menu key, folder under weights/minim, body parts, modality)
_VARIANTS = [
    ("oct", "oct", "Retina (eye)", "Optical coherence tomography"),
    ("fundus", "fundus", "Retina (eye)", "Fundus photography"),
    ("chest-xray", "chest-xray", "Chest / lungs", "Chest X-ray"),
    ("chest-ct", "mri-ct", "Chest / lungs", "Chest CT (2D slice)"),
    ("brain-mri", "mri-ct", "Brain", "Brain MRI (2D slice)"),
    ("breast-mri", "mri-ct", "Breast", "Breast MRI (2D slice)"),
]

MODELS = [
    Model(
        key=f"minim-{key}",
        name=f"MINIM {key}",
        body_parts=body_parts,
        modalities=modality,
        generate=partial(generate, folder),
    )
    for key, folder, body_parts, modality in _VARIANTS
]
