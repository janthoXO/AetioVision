"""Supported set (D13), derived at startup from catalog + masker + inpainter capabilities.

A row (modality, site, morphology) is supported when a template covers the site and an
inpainter capability (modality, scope, morphology) has scope among the site's ancestors.
Templates cover a site through a predefined mask for it, or (open-vocabulary masker only)
through their region being an ancestor of the site. First inpainter wins (D27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aetiovision.domain.codes import Code, Triple
from aetiovision.pipeline.ports import Inpainter, Terminology


@dataclass(frozen=True)
class Row:
    modality: Code
    site: Code  # mask site, or template region when open
    morphology: Code
    open: bool  # True: any site within `site` (region), masker finds it
    scope: Code  # capability scope, must be an ancestor of the actual site


@dataclass
class SupportedSet:
    terminology: Terminology
    masks: set[tuple[Code, Code]]  # (modality, mask site)
    regions: set[tuple[Code, Code]]  # (modality, region), empty without open-vocabulary masker
    inpainters: list[Inpainter]
    _anc: dict[Code, list[Code]] = field(default_factory=dict, repr=False)

    def ancestors(self, code: Code) -> list[Code]:
        if code not in self._anc:
            self._anc[code] = self.terminology.ancestors(code)
        return self._anc[code]

    def caps(self) -> list[tuple[Inpainter, Triple]]:
        return [(i, c) for i in self.inpainters for c in sorted(i.capabilities(), key=str)]

    def rows(self) -> list[Row]:
        out, seen = [], set()
        coverage = [(m, s, False) for m, s in self.masks] + [(m, r, True) for m, r in self.regions]
        for m, s, open_ in sorted(coverage, key=str):
            anc = self.ancestors(s)
            for _, cap in self.caps():
                # open region: scope is checked against the actual site in matching()
                key = (m, s, cap.morphology, open_, cap.body_site)
                if cap.modality == m and (open_ or cap.body_site in anc) and key not in seen:
                    seen.add(key)
                    out.append(Row(m, s, cap.morphology, open_, cap.body_site))
        return out

    # --- roll up (resolution rule 6) ---

    def site_for(self, modality: Code, site: Code) -> Code | None:
        """Nearest ancestor-or-self of site with a predefined mask, or inside a template region."""
        for a in self.ancestors(site):
            if (modality, a) in self.masks:
                return a
            if any((modality, r) in self.regions for r in self.ancestors(a)):
                return a
        return None

    def morphology_for(self, modality: Code, site: Code, morphology: Code) -> Code | None:
        site_anc = set(self.ancestors(site))
        for a in self.ancestors(morphology):
            if any(c.modality == modality and c.body_site in site_anc and c.morphology == a for _, c in self.caps()):
                return a
        return None

    def rollup(self, t: Triple) -> Triple | None:
        site = self.site_for(t.modality, t.body_site)
        morph = site and self.morphology_for(t.modality, site, t.morphology)
        return Triple(t.modality, site, morph) if morph else None

    def inpainter_for(self, t: Triple) -> Inpainter:
        site_anc = set(self.ancestors(t.body_site))
        for inp, c in self.caps():
            if c.modality == t.modality and c.body_site in site_anc and c.morphology == t.morphology:
                return inp
        raise LookupError(f"no inpainter for {t.label()}")  # rollup() guarantees one

    def regions_for(self, t: Triple) -> set[Code]:
        """Template regions that contain the (supported) site."""
        anc = set(self.ancestors(t.body_site))
        return {r for m, r in self.regions if m == t.modality and r in anc}

    # --- gap filling (D19) ---

    def matching(self, modality: Code | None, site: Code | None, morphology: Code | None) -> list[Row]:
        """Rows compatible with the given (possibly missing) slots."""
        out = []
        for row in self.rows():
            if modality and row.modality != modality:
                continue
            if site:
                anc = self.ancestors(site)
                if row.open and (row.site not in anc or row.scope not in anc):
                    continue
                if not row.open and self.site_for(row.modality, site) != row.site:
                    continue
            if morphology and row.morphology not in self.ancestors(morphology):
                continue
            out.append(row)
        return out

    def listing(self) -> list[str]:
        lookup = self.terminology.lookup
        return [
            f"{r.modality} / "
            + (f"any {lookup(r.scope).display} within " if r.open else "")
            + f"{lookup(r.site).label()} / {lookup(r.morphology).label()}"
            for r in self.rows()
        ]
