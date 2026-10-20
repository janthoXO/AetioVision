"""All ports (D2, D3). Adapters implement these structurally, no inheritance needed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aetiovision.domain.codes import Code, Triple
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.request import CodeRequest, ResolvedRequest
from aetiovision.domain.template import Template

if TYPE_CHECKING:
    from aetiovision.pipeline.supported import SupportedSet


@dataclass(frozen=True)
class ProcedureParts:
    modality: Code | None  # DCM
    body_site: Code | None  # SCT, imaging focus wins over region imaged
    laterality: Code | None  # SCT


class Terminology(Protocol):
    def lookup(self, code: Code) -> Code: ...  # display filled, UnknownCode if absent / inactive
    def ancestors(self, code: Code) -> list[Code]: ...  # self first, nearest first (IS-A + Part of)
    def finding_groups(self, finding: Code) -> list[tuple[Code | None, Code | None]]: ...  # [(site, morphology)] per role group
    def split_laterality(self, site: Code) -> tuple[Code, Code | None]: ...  # left femur -> (femur, left)
    def procedure_parts(self, loinc: Code) -> ProcedureParts: ...  # UnknownCode if part unmapped
    def find(self, term: str, root: str) -> list[Code]: ...  # case-insensitive exact description match under root
    def search(self, text: str, root: str, limit: int = 10) -> list[Code]: ...  # substring match under root, shortest term first


class CodeExtractor(Protocol):
    def extract(self, text: str, supported: SupportedSet) -> CodeRequest: ...


class TemplateCatalog(Protocol):
    def coverage(self) -> set[tuple[Code, Code]]: ...  # (modality, predefined mask body site)
    def regions(self) -> set[tuple[Code, Code]]: ...  # (modality, template region)

    def candidates(self, req: ResolvedRequest, regions: set[Code]) -> list[Template]:
        """Templates with modality and laterality (template laterality NULL matches any).
        Ones with a predefined mask for req.supported.body_site first, then ones whose
        region is in `regions` (empty when no fallback masker). Each group seeded-shuffled."""
        ...


class AnatomyMasker(Protocol):
    def open_vocabulary(self) -> bool: ...  # True when it can segment sites without predefined mask
    def mask(self, template: Template, site: Code, laterality: Code | None) -> Mask: ...  # MaskingFailed


class Inpainter(Protocol):
    name: str

    def capabilities(self) -> set[Triple]: ...  # abstract codes: (DX, Bone structure, Fracture)
    def inpaint(self, template: Template, mask: Mask, req: ResolvedRequest) -> MedicalImage: ...  # InpaintFailed
