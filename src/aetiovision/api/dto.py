from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from aetiovision.domain.codes import Code
from aetiovision.domain.request import CodeRequest

DCM = Annotated[str, StringConstraints(pattern=r"^DCM:[A-Z]{2}$")]
SCT = Annotated[str, StringConstraints(pattern=r"^SCT:\d{6,18}$")]
LOINC = Annotated[str, StringConstraints(pattern=r"^LOINC:\d{1,7}-\d$")]


class GenerateFromCodes(BaseModel):
    """Give a triple, a SNOMED finding, a LOINC procedure, or a mix. Missing slots
    are filled from the other codes or inferred when only one supported value fits."""

    model_config = ConfigDict(extra="forbid")

    # canonical triple, each slot optional when another code supplies it
    modality: DCM | None = Field(None, description="DICOM modality: DCM:DX (X-ray), DCM:CT, DCM:MR")
    body_site: SCT | None = Field(None, description="SNOMED CT body structure, e.g. SCT:41111004 shaft of femur")
    morphology: SCT | None = Field(None, description="SNOMED CT morphologic abnormality, e.g. SCT:72704001 Fracture")
    laterality: SCT | None = Field(None, description="SCT:7771000 left, SCT:24028007 right, SCT:51440002 bilateral")

    # pre-coordinated sources, resolved to triple slots
    finding: SCT | None = Field(None, description="SNOMED CT clinical finding, e.g. SCT:71620000 Fracture of femur")
    procedure: LOINC | None = Field(None, description="LOINC radiology term, e.g. LOINC:36654-2 XR Femur - left 2 Views")

    instructions: str = Field("", max_length=2000, description="free detail for the inpainter prompt")
    seed: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _some_code(self) -> GenerateFromCodes:
        if not any((self.modality, self.body_site, self.morphology, self.finding, self.procedure)):
            raise ValueError("give a triple, a SNOMED finding, a LOINC procedure, or a mix")
        return self

    def to_domain(self) -> CodeRequest:
        def p(v: str | None) -> Code | None:
            return Code.parse(v) if v else None

        return CodeRequest(
            modality=p(self.modality),
            body_site=p(self.body_site),
            morphology=p(self.morphology),
            laterality=p(self.laterality),
            finding=p(self.finding),
            procedure=p(self.procedure),
            instructions=self.instructions,
            seed=self.seed,
        )
