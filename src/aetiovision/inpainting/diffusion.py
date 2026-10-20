"""Diffusion inpainter (D29): crops around the mask, inpaints in a worker process, pastes the crop back."""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from aetiovision.domain.codes import BODY_STRUCTURE, LEFT, RIGHT, Code, Triple
from aetiovision.domain.errors import InpaintFailed
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.template import Template
from aetiovision.shared.worker import WorkerError, call

MODALITY_TEXT = {"DX": "x-ray radiograph", "CT": "CT scan", "MR": "MRI scan"}
NEGATIVE = "text, letters, watermark, annotation, marker, drawing, cartoon, blurry, low quality, artifacts"
MORPHOLOGIES = ["72704001", "108369006"]  # Fracture, Neoplasm


def build_prompt(req: ResolvedRequest) -> str:
    """MediSyn is caption-trained: a plain descriptive sentence, e.g. 'X-ray of the left femur showing a transverse fracture'."""
    mod = MODALITY_TEXT.get(req.requested.modality.code, "medical image")
    site = (req.requested.body_site.display or "body part").lower()
    morph = (req.requested.morphology.display or "abnormality").lower()
    side = {LEFT: "left ", RIGHT: "right "}.get(req.laterality.code if req.laterality else "", "")
    parts = [f"{mod} of the {side}{site} showing a {morph}"]
    if req.instructions.strip():
        parts.append(req.instructions.strip())
    return ", ".join(parts)


def crop_box(alpha: np.ndarray, margin: float = 0.3) -> tuple[int, int, int, int]:
    """Square (y0, y1, x0, x1) around the alpha>0.05 bbox with margin, clamped to the image."""
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0.05)
    if len(ys) == 0:
        raise InpaintFailed("empty mask")
    side = min(int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1 + 2 * margin)), h, w)
    cy, cx = (ys.min() + ys.max() + 1) // 2, (xs.min() + xs.max() + 1) // 2
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return y0, y0 + side, x0, x0 + side


def paste(image: np.ndarray, crop: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Copy of image with crop (resized to the box if needed) written into box."""
    y0, y1, x0, x1 = box
    if crop.shape != (y1 - y0, x1 - x0):
        crop = np.asarray(Image.fromarray(crop).resize((x1 - x0, y1 - y0), Image.BICUBIC))
    out = image.copy()
    out[y0:y1, x0:x1] = crop
    return out


def _resize(a: np.ndarray, size: int, resample) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((size, size), resample))


class DiffusionInpainter:
    name = "diffusion"

    def __init__(self, url: str = "http://127.0.0.1:8402", steps: int = 30, size: int = 512):
        self.url, self.steps, self.size = url, steps, size

    def capabilities(self) -> set[Triple]:
        return {Triple(Code("DCM", m), Code.sct(BODY_STRUCTURE), Code.sct(c)) for m in ("DX", "CT", "MR") for c in MORPHOLOGIES}

    def inpaint(self, template: Template, mask: Mask, req: ResolvedRequest) -> MedicalImage:
        img = np.rint(np.clip(template.image.pixels, 0, 1) * 255).astype(np.uint8)
        box = crop_box(mask.alpha)
        y0, y1, x0, x1 = box
        crop = _resize(img[y0:y1, x0:x1], self.size, Image.BICUBIC)
        m = _resize(((mask.alpha[y0:y1, x0:x1] > 0.5) * 255).astype(np.uint8), self.size, Image.NEAREST) > 127
        if not m.any():
            raise InpaintFailed("mask empty at inpainting resolution")  # else the template comes back unchanged as a "result"
        m = (binary_dilation(m, iterations=max(1, self.size // 100)) * 255).astype(np.uint8)
        payload = {
            "image": crop, "mask": m, "prompt": build_prompt(req), "negative_prompt": NEGATIVE,
            "seed": req.seed, "steps": self.steps, "guidance": 7.5, "strength": 0.85,
        }
        try:
            out = call(self.url, "inpaint", payload, spawn=[sys.executable, "-m", "aetiovision.main", "worker", "diffusion"])["image"]
        except WorkerError as e:
            raise InpaintFailed(str(e)) from e
        full = paste(img, out.astype(np.uint8), box)
        return MedicalImage(full.astype(np.float32) / 255, {"inpainter": self.name, "prompt": payload["prompt"], "box": box, "steps": self.steps})
