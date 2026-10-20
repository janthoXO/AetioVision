from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MedicalImage:
    pixels: np.ndarray  # float32 HxW in [0, 1], windowed at ingestion (D17)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Mask:
    alpha: np.ndarray  # float32 HxW in [0, 1], soft (Gaussian feathered, D26)
    source: str = ""  # "predefined:<path>" or "grounded:<model>"
