"""In-memory fakes per port. FakeTerminology reads tests/mock_terminology.json (real SNOMED/LOINC excerpt)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aetiovision.domain.codes import BILATERAL, IS_A, LATERALITY, LEFT, PART_OF, RIGHT, Code, Triple
from aetiovision.domain.errors import MaskingFailed, UnknownCode
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.template import Template
from aetiovision.pipeline.ports import ProcedureParts

MOCK = json.loads(Path(__file__).with_name("mock_terminology.json").read_text())
RAD_MODALITY = {"XR": "DX", "CT": "CT", "MR": "MR"}
SIDES = {LEFT, RIGHT, BILATERAL}


def _name(fsn: str) -> str:
    return re.sub(r" \([^)]*\)$", "", fsn)


class FakeTerminology:
    def __init__(self, data: dict = MOCK):
        self.sct, self.loinc, self.part_sct = data["sct"], data["loinc"], data["loinc_part_sct"]

    def lookup(self, code: Code) -> Code:
        if code.system == "SCT":
            if code.code not in self.sct:
                raise UnknownCode(f"unknown SNOMED code {code}")
            return Code("SCT", code.code, _name(self.sct[code.code]["fsn"]))
        if code.system == "LOINC":
            if code.code not in self.loinc:
                raise UnknownCode(f"unknown LOINC code {code}")
            return Code("LOINC", code.code, self.loinc[code.code]["name"])
        return code

    def ancestors(self, code: Code) -> list[Code]:
        if code.system != "SCT":
            return [code]
        out, frontier = [code.code], [code.code]
        while frontier:  # BFS: nearest first
            nxt = []
            for c in frontier:
                c = self.sct[c]
                for p in c["parents"] + [v for _, t, v in c["attrs"] if t == PART_OF]:
                    if p not in out:
                        out.append(p)
                        nxt.append(p)
            frontier = nxt
        return [self.lookup(Code.sct(c)) for c in out]

    def finding_groups(self, finding: Code) -> list[tuple[Code | None, Code | None]]:
        groups: dict[int, dict[str, str]] = {}
        for g, t, v in self.sct[finding.code]["attrs"]:
            groups.setdefault(g, {})[t] = v
        out = []
        for g in sorted(groups):
            s, m = groups[g].get("363698007"), groups[g].get("116676008")
            if s or m:
                out.append((s and self.lookup(Code.sct(s)), m and self.lookup(Code.sct(m))))
        return out

    def split_laterality(self, site: Code) -> tuple[Code, Code | None]:
        c = self.sct[site.code]
        lat = next((v for _, t, v in c["attrs"] if t == LATERALITY and v in SIDES), None)
        if not lat:
            return site, None
        for p in c["parents"]:
            if not any(t == LATERALITY and v in SIDES for _, t, v in self.sct[p]["attrs"]):
                return self.lookup(Code.sct(p)), self.lookup(Code.sct(lat))
        return site, None

    def procedure_parts(self, loinc: Code) -> ProcedureParts:
        if loinc.code not in self.loinc:
            raise UnknownCode(f"unknown LOINC code {loinc}")
        parts = {t: (num, name) for t, num, name in self.loinc[loinc.code]["parts"]}

        def sct(part_type):
            if part_type not in parts:
                return None
            num, name = parts[part_type]
            if num not in self.part_sct:
                raise UnknownCode(f"LOINC part {num} {name} has no SNOMED mapping")
            return self.lookup(Code.sct(self.part_sct[num]))

        mod = parts.get("Rad.Modality.Modality Type")
        return ProcedureParts(
            modality=Code("DCM", RAD_MODALITY[mod[1]]) if mod and mod[1] in RAD_MODALITY else None,
            body_site=sct("Rad.Anatomic Location.Imaging Focus") or sct("Rad.Anatomic Location.Region Imaged"),
            laterality=sct("Rad.Anatomic Location.Laterality"),
        )

    def _isa_under(self, cid: str, root: str) -> bool:
        seen, todo = set(), [cid]
        while todo:
            c = todo.pop()
            if c == root:
                return True
            if c not in seen:
                seen.add(c)
                todo += self.sct[c]["parents"]
        return False

    def _terms(self, cid: str) -> list[str]:
        c = self.sct[cid]
        return [_name(c["fsn"]), *c["synonyms"]]

    def find(self, term: str, root: str) -> list[Code]:
        t = term.lower()
        return [self.lookup(Code.sct(c)) for c in sorted(self.sct) if self._isa_under(c, root) and t in (x.lower() for x in self._terms(c))]

    def search(self, text: str, root: str, limit: int = 10) -> list[Code]:
        t = text.lower()
        hits = [(min(len(x) for x in self._terms(c) if t in x.lower()), c) for c in self.sct
                if self._isa_under(c, root) and any(t in x.lower() for x in self._terms(c))]
        return [self.lookup(Code.sct(c)) for _, c in sorted(hits)[:limit]]


def blank_template(id="t1", modality="DX", region="61685007", masks=None, laterality=None) -> Template:
    pixels = np.full((64, 64), 0.5, dtype=np.float32)
    return Template(id, MedicalImage(pixels), Code("DCM", modality), Code.sct(region),
                    laterality and Code.sct(laterality), "AP", "fake", "CC0", masks or {})


@dataclass
class FakeCatalog:
    templates: list[Template] = field(default_factory=lambda: [blank_template(masks={"SCT:41111004": Path("m.png")})])

    def coverage(self):
        return {(t.modality, Code.parse(s)) for t in self.templates for s in t.masks}

    def regions(self):
        return {(t.modality, t.region) for t in self.templates}

    def candidates(self, req, regions):
        site, lat = str(req.supported.body_site), req.laterality
        ok = [t for t in self.templates if t.modality == req.supported.modality and (t.laterality is None or t.laterality == lat)]
        first = [t for t in ok if site in t.masks]
        rest = [t for t in ok if t not in first and t.region in regions]
        rng = random.Random(req.seed)
        rng.shuffle(first)
        rng.shuffle(rest)
        return first + rest


@dataclass
class FakeMasker:
    open: bool = True
    fail_ids: tuple = ()
    calls: list = field(default_factory=list)

    def open_vocabulary(self):
        return self.open

    def mask(self, template, site, laterality):
        self.calls.append(template.id)
        if template.id in self.fail_ids:
            raise MaskingFailed("not found")
        a = np.zeros(template.image.pixels.shape, np.float32)
        a[16:48, 24:40] = 1
        return Mask(a, "fake")


@dataclass
class FakeInpainter:
    name: str = "fake"
    caps: set = field(default_factory=lambda: {Triple(Code("DCM", "DX"), Code.sct("272673000"), Code.sct("72704001"))})

    def capabilities(self):
        return self.caps

    def inpaint(self, template, mask, req):
        return MedicalImage(np.ones_like(template.image.pixels))  # paints everywhere: pipeline must clip to mask


@dataclass
class FakeExtractor:
    request: object = None

    def extract(self, text, supported):
        return self.request
