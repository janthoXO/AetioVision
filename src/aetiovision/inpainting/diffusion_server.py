"""Diffusion worker: SD 1.x inpainting pipeline, optionally with the MediSyn UNet (4-channel, latent blending)."""

from __future__ import annotations

import os

import numpy as np

from aetiovision.shared.worker import serve


def main() -> None:
    import torch
    from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
    from PIL import Image

    base = os.environ.get("INPAINTING_BASE", "stable-diffusion-v1-5/stable-diffusion-inpainting")
    unet_id = os.environ.get("INPAINTING_UNET", "")  # "hiesingerlab/MediSyn" with base stable-diffusion-v1-5/stable-diffusion-v1-5 for the 4-channel path
    port = int(os.environ.get("INPAINTING_PORT", "8402"))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kw = {"unet": UNet2DConditionModel.from_pretrained(unet_id, torch_dtype=dtype)} if unet_id else {}
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base, safety_checker=None, requires_safety_checker=False, torch_dtype=dtype, **kw
    ).to(device)
    if device == "mps":
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    def inpaint(p: dict) -> dict:
        rgb = Image.fromarray(p["image"]).convert("RGB")
        mask = Image.fromarray(p["mask"]).convert("L")
        gen = torch.Generator("cpu").manual_seed(int(p["seed"]))
        out = pipe(
            prompt=p["prompt"], negative_prompt=p.get("negative_prompt"), image=rgb, mask_image=mask, generator=gen,
            num_inference_steps=int(p["steps"]), guidance_scale=float(p["guidance"]), strength=float(p["strength"]),
            height=rgb.height, width=rgb.width,
        ).images[0]
        return {"image": np.asarray(out.convert("L"), dtype=np.uint8)}

    serve({"inpaint": inpaint}, port)


if __name__ == "__main__":
    main()
