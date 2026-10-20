"""Predefined mask PNG if the template has one for the site, else the fallback masker; then QA and feather (D26)."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label

from aetiovision.domain.codes import Code
from aetiovision.domain.errors import MaskingFailed
from aetiovision.domain.image import Mask
from aetiovision.domain.template import Template
from aetiovision.pipeline.ports import AnatomyMasker

MIN_AREA, MAX_AREA, MIN_COMPONENT = 0.0005, 0.6, 0.1


def qa(binary: np.ndarray) -> np.ndarray:
    """Area fraction in [MIN_AREA, MAX_AREA]; drops components smaller than 10% of the largest."""
    binary = binary.astype(bool)
    frac = binary.mean()
    if not binary.any() or not MIN_AREA <= frac <= MAX_AREA:
        raise MaskingFailed(f"mask QA failed: area fraction {frac:.4f} outside [{MIN_AREA}, {MAX_AREA}]")
    comps, n = label(binary)
    sizes = np.bincount(comps.ravel())[1:]
    keep = np.flatnonzero(sizes >= MIN_COMPONENT * sizes.max()) + 1
    return np.isin(comps, keep)


def feather(binary: np.ndarray, sigma: float) -> np.ndarray:
    """Soft edge outward only: the mask itself stays 1, so thin structures (fingers, ribs) don't blur away."""
    b = binary.astype(np.float32)
    return np.clip(np.maximum(b, gaussian_filter(b, sigma)), 0, 1).astype(np.float32)


class PredefinedMasker:
    def __init__(self, fallback: AnatomyMasker | None = None, sigma_frac: float = 0.01):
        self.fallback, self.sigma_frac = fallback, sigma_frac

    def open_vocabulary(self) -> bool:
        return self.fallback is not None

    def mask(self, template: Template, site: Code, laterality: Code | None) -> Mask:
        h, w = template.image.pixels.shape
        if str(site) in template.masks:
            path = template.masks[str(site)]
            png = Image.open(path).convert("L")
            if png.size != (w, h):
                png = png.resize((w, h), Image.NEAREST)
            binary, source = np.asarray(png, np.float32) / 255 > 0.5, f"predefined:{path}"
        elif self.fallback is not None:
            m = self.fallback.mask(template, site, laterality)
            binary, source = qa(m.alpha > 0.5), m.source
        else:
            raise MaskingFailed(f"template {template.id} has no predefined mask for {site.label()} and no fallback masker")
        return Mask(feather(binary, self.sigma_frac * max(h, w)), source)
