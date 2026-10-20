from __future__ import annotations

from dataclasses import dataclass

from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.template import Template


@dataclass(frozen=True)
class Result:
    resolved: ResolvedRequest
    template: Template
    mask: Mask
    image: MedicalImage
    inpainter: str
