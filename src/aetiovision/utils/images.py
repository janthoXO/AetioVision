"""Image IO and result serialisation. Pixels are float32 HxW in [0, 1]."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from aetiovision.domain.codes import Code
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.result import Result


def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16", "I"):
        return np.asarray(im, dtype=np.float32) / 65535
    return np.asarray(im.convert("L"), dtype=np.float32) / 255


def _img(pixels: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8), "L")


def save_gray(pixels: np.ndarray, path: Path) -> None:
    _img(pixels).save(path, format="PNG")


def preview_png(pixels: np.ndarray, max_side: int = 512) -> bytes:
    im = _img(pixels)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _code(c: Code | None) -> dict | None:
    return c and {"code": str(c), "display": c.display}


def _triple(t) -> dict:
    return {
        "modality": str(t.modality), "modality_display": t.modality.display,
        "body_site": str(t.body_site), "body_site_display": t.body_site.display,
        "morphology": str(t.morphology), "morphology_display": t.morphology.display,
    }


def resolved_to_dict(resolved: ResolvedRequest) -> dict:
    return {
        "requested": _triple(resolved.requested),
        "supported": _triple(resolved.supported),
        "laterality": _code(resolved.laterality),
        "instructions": resolved.instructions,
        "seed": resolved.seed,
        "sources": dict(resolved.sources),
        "assumptions": list(resolved.assumptions),
    }


def save_result(result: Result, out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t, seed = result.template, result.resolved.seed
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{t.id}_{seed}"
    paths = {k: (out_dir / f"{stem}{s}").resolve() for k, s in
             [("image", ".png"), ("mask", "_mask.png"), ("template", "_template.png"), ("json", ".json")]}
    save_gray(result.image.pixels, paths["image"])
    save_gray(result.mask.alpha, paths["mask"])
    save_gray(t.image.pixels, paths["template"])
    paths = {k: str(v) for k, v in paths.items()}
    meta = resolved_to_dict(result.resolved) | {
        "template": {"id": t.id, "source": t.source, "license": t.license},
        "inpainter": result.inpainter,
        "mask": {"source": result.mask.source},
        "files": paths,
    }
    Path(paths["json"]).write_text(json.dumps(meta, indent=2))
    return paths
