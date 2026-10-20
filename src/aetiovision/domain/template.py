from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aetiovision.domain.codes import Code
from aetiovision.domain.image import MedicalImage


@dataclass(frozen=True)
class Template:
    id: str
    image: MedicalImage
    modality: Code
    region: Code  # SNOMED body structure the image shows
    laterality: Code | None
    view: str | None
    source: str
    license: str
    masks: dict[str, Path] = field(default_factory=dict)  # "SCT:<site>" -> predefined mask PNG
