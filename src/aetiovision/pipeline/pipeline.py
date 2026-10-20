from __future__ import annotations

from dataclasses import dataclass, field, replace

from aetiovision.domain.errors import MaskingFailed, NoTemplate
from aetiovision.domain.image import MedicalImage
from aetiovision.domain.request import CodeRequest, ResolvedRequest
from aetiovision.domain.result import Result
from aetiovision.pipeline.ports import AnatomyMasker, CodeExtractor, Inpainter, TemplateCatalog, Terminology
from aetiovision.pipeline.resolution import resolve
from aetiovision.pipeline.supported import SupportedSet

MAX_TEMPLATE_TRIES = 3  # model masking may not find the site on a region template; try the next one


@dataclass
class Pipeline:
    terminology: Terminology
    extractor: CodeExtractor
    catalog: TemplateCatalog
    masker: AnatomyMasker
    inpainters: list[Inpainter]  # order = priority (D27)
    supported: SupportedSet = field(init=False)

    def __post_init__(self) -> None:
        regions = self.catalog.regions() if self.masker.open_vocabulary() else set()
        self.supported = SupportedSet(self.terminology, self.catalog.coverage(), regions, self.inpainters)

    def extract(self, text: str, seed: int | None = None) -> CodeRequest:
        return replace(self.extractor.extract(text, self.supported), seed=seed)

    def resolve(self, req: CodeRequest) -> ResolvedRequest:
        return resolve(req, self.terminology, self.supported)

    def from_text(self, text: str, seed: int | None = None) -> Result:
        return self.from_codes(self.extract(text, seed))

    def from_codes(self, req: CodeRequest) -> Result:
        resolved = self.resolve(req)
        site = resolved.supported.body_site
        templates = self.catalog.candidates(resolved, self.supported.regions_for(resolved.supported))
        if not templates:
            raise NoTemplate(f"no {resolved.supported.modality} template covers {site.label()}")
        errors = []
        for template in templates[:MAX_TEMPLATE_TRIES]:
            try:
                mask = self.masker.mask(template, site, resolved.laterality)
                break
            except MaskingFailed as e:
                errors.append(f"{template.id}: {e}")
        else:
            raise MaskingFailed("; ".join(errors))
        inpainter = self.supported.inpainter_for(resolved.supported)
        out = inpainter.inpaint(template, mask, resolved)
        # leakage invariant: nothing outside the soft mask changes, whatever the inpainter did
        a, src = mask.alpha, template.image.pixels
        pixels = (src * (1 - a) + out.pixels.clip(0, 1) * a).astype("float32")
        return Result(resolved, template, mask, MedicalImage(pixels, out.meta), inpainter.name)
