"""Fallback masker: Qwen3-VL (Ollama) grounds the site name to boxes, MedSAM (worker process) turns them into a mask."""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from aetiovision.domain.codes import BILATERAL, LEFT, RIGHT, Code
from aetiovision.domain.errors import MaskingFailed
from aetiovision.domain.image import Mask
from aetiovision.domain.template import Template
from aetiovision.shared import worker

MODALITY_NAMES = {"DX": "X-ray radiograph", "CT": "CT slice", "MR": "MR image"}
SPAWN = [sys.executable, "-m", "aetiovision.main", "worker", "medsam"]
PROMPT = (
    "Locate the {site} in this {modality}. Answer only with JSON: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}] with coordinates normalized to 0-1000. '
    "Answer [] if it is not visible."
)


def anatomy_name(site: Code) -> str:
    """'Bone structure of shaft of femur' -> 'shaft of femur': SNOMED prefixes confuse the VLM's box extent."""
    return re.sub(r"^(Bone structure|Structure|Entire|Skeletal structure) of ", "", site.display) or str(site)


def parse_boxes(text: str, w: int, h: int) -> list[list[float]]:
    """VLM answer (maybe fenced, list or dict, 0-1000 coords) -> pixel boxes, clamped, degenerate dropped."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        try:
            data = json.loads(m.group()) if m else []
        except ValueError:
            return []
    if isinstance(data, dict):
        data = [data]
    boxes = []
    for item in data if isinstance(data, list) else []:
        b = item.get("bbox_2d") or item.get("bbox") if isinstance(item, dict) else None
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((min(max(x1, 0), 1000) * w / 1000, min(max(x2, 0), 1000) * w / 1000))
        y1, y2 = sorted((min(max(y1, 0), 1000) * h / 1000, min(max(y2, 0), 1000) * h / 1000))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append([x1, y1, x2, y2])
    return boxes


def choose_boxes(boxes: list[list[float]], laterality: Code | None, w: int) -> list[list[float]]:
    """VLM left/right labels are unreliable: radiological convention, patient's right is on the image left."""
    if len(boxes) <= 1:
        return boxes
    cx = lambda b: b[0] + b[2]
    code = laterality.code if laterality else None
    if code == RIGHT:
        return [min(boxes, key=cx)]
    if code == LEFT:
        return [max(boxes, key=cx)]
    if code == BILATERAL:
        return boxes
    return [max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]


class GroundedMasker:
    def __init__(self, vlm_model: str = "qwen3-vl:8b-instruct", ollama_url: str = "http://localhost:11434", medsam_url: str = "http://127.0.0.1:8401"):
        self.vlm_model, self.ollama_url, self.medsam_url = vlm_model, ollama_url, medsam_url

    def open_vocabulary(self) -> bool:
        return True

    def mask(self, template: Template, site: Code, laterality: Code | None) -> Mask:
        img = (np.clip(template.image.pixels, 0, 1) * 255).round().astype(np.uint8)
        h, w = img.shape
        boxes = choose_boxes(self._ground(img, site, MODALITY_NAMES.get(template.modality.code, "medical image")), laterality, w)
        if not boxes:
            raise MaskingFailed(f"{self.vlm_model} did not find {site.label()} in template {template.id}")
        try:
            masks = worker.call(self.medsam_url, "segment", {"image": img, "boxes": boxes}, spawn=SPAWN)["masks"]
        except (urllib.error.URLError, worker.WorkerError, ValueError, KeyError) as e:
            raise MaskingFailed(f"MedSAM worker failed: {e}") from e
        return Mask((masks > 0).any(axis=0).astype(np.float32), f"grounded:{self.vlm_model}+medsam")

    def _ground(self, img: np.ndarray, site: Code, modality: str) -> list[list[float]]:
        h, w = img.shape
        pil = Image.fromarray(img)
        pil.thumbnail((1024, 1024))
        buf = io.BytesIO()
        pil.save(buf, "PNG")
        body = {
            "model": self.vlm_model,
            "stream": False,
            "messages": [{"role": "user", "content": PROMPT.format(site=anatomy_name(site), modality=modality), "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(f"{self.ollama_url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            text = json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            raise MaskingFailed(f"Ollama request to {self.ollama_url} failed: {e}") from e
        return parse_boxes(text, w, h)
