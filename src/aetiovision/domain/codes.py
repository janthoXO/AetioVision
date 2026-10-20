from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aetiovision.domain.errors import InvalidCode

System = Literal["DCM", "SCT", "LOINC"]

# SNOMED CT constants (D12: everything else comes from the release files)
BODY_STRUCTURE = "123037004"
MORPHOLOGY = "49755003"
CLINICAL_FINDING = "404684003"
IS_A = "116680003"
FINDING_SITE = "363698007"
ASSOCIATED_MORPHOLOGY = "116676008"
LATERALITY = "272741003"
PART_OF = "123005000"
LEFT, RIGHT, BILATERAL = "7771000", "24028007", "51440002"

MODALITIES = {"DX", "CR", "CT", "MR"}


@dataclass(frozen=True)
class Code:
    system: System
    code: str
    display: str = field(default="", compare=False)

    @classmethod
    def parse(cls, curie: str) -> Code:
        system, sep, code = curie.strip().partition(":")
        if not sep or system not in ("DCM", "SCT", "LOINC") or not code:
            raise InvalidCode(f"not a DCM:/SCT:/LOINC: code: {curie!r}")
        if system == "DCM":
            if code not in MODALITIES:
                raise InvalidCode(f"unsupported DICOM modality {code!r}, use one of {sorted(MODALITIES)}")
            code = "DX" if code == "CR" else code  # CR treated as DX
        return cls(system, code)

    @classmethod
    def sct(cls, code: str, display: str = "") -> Code:
        return cls("SCT", code, display)

    def __str__(self) -> str:
        return f"{self.system}:{self.code}"

    def label(self) -> str:
        return f"{self} {self.display}".strip()


@dataclass(frozen=True)
class Triple:
    modality: Code  # DCM
    body_site: Code  # SCT body structure
    morphology: Code  # SCT morphologic abnormality

    def label(self) -> str:
        return " / ".join(c.label() for c in (self.modality, self.body_site, self.morphology))
