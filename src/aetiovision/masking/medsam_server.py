"""MedSAM worker: box prompts -> masks. Run as `aetio worker medsam`."""

from __future__ import annotations

import os

import numpy as np

from aetiovision.shared import worker


def main() -> None:
    import torch
    from transformers import SamModel, SamProcessor

    name = os.environ.get("MASKING_MODEL", "wanglab/medsam-vit-base")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(name).to(device).eval()
    processor = SamProcessor.from_pretrained(name)

    def segment(req: dict) -> dict:
        from PIL import Image

        rgb = Image.fromarray(np.asarray(req["image"], np.uint8)).convert("RGB")
        masks = []
        for box in req["boxes"]:
            inputs = processor(rgb, input_boxes=[[[float(v) for v in box]]], return_tensors="pt")
            inputs = {k: v.float().to(device) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}  # MPS has no float64
            with torch.no_grad():
                out = model(**inputs, multimask_output=False)
            m = processor.image_processor.post_process_masks(out.pred_masks.sigmoid().cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(), binarize=False)[0]
            masks.append((m[0, 0] > 0.5).numpy().astype(np.uint8))
        return {"masks": np.stack(masks)}

    worker.serve({"segment": segment}, int(os.environ.get("MASKING_PORT") or 8401))


if __name__ == "__main__":
    main()
