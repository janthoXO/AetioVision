"""Terminology adapter over terminology.db (see importer.py). Same semantics as tests/fakes.FakeTerminology."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from aetiovision.domain.codes import (
    ASSOCIATED_MORPHOLOGY,
    BILATERAL,
    BODY_STRUCTURE,
    FINDING_SITE,
    LATERALITY,
    LEFT,
    PART_OF,
    RIGHT,
    Code,
)
from aetiovision.domain.errors import UnknownCode
from aetiovision.pipeline.ports import ProcedureParts

RAD_MODALITY = {"XR": "DX", "CT": "CT", "MR": "MR"}  # hand-written LOINC mappings: these two
RAD_SIDE = {"left": LEFT, "right": RIGHT, "bilateral": BILATERAL}  # "Unspecified", "Unilateral" -> no laterality
SIDES = (LEFT, RIGHT, BILATERAL)
UP = "WITH RECURSIVE u(id) AS (SELECT ? UNION SELECT parent FROM sct_isa JOIN u ON child = u.id) SELECT id FROM u"


def _name(fsn: str) -> str:
    return re.sub(r" \([^)]*\)$", "", fsn)


def _singular(word: str) -> str:
    return word[:-3] + "y" if word.endswith("ies") else word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def part_terms(name: str) -> Iterator[str]:
    """LOINC Rad part name -> SNOMED term guesses: 'Spine.lumbar' -> 'lumbar spine', 'Ribs' -> 'rib', 'Hip' -> 'hip region'.
    Qualifiers are dropped from the right when nothing matches ('Thigh.soft tissue' -> 'thigh'): coarser, still containing."""
    segs = name.lower().split(".")
    for k in range(len(segs), 0, -1):
        term = " ".join(reversed(segs[:k]))
        one = " ".join(_singular(w) for w in term.split())
        yield from dict.fromkeys((term, one, one + " region", one + " structure"))


class SqliteTerminology:
    def __init__(self, db: Path):
        if not Path(db).exists():
            raise FileNotFoundError(f"terminology.db not found at {db}, run `aetio import-terminology --snomed DIR --loinc DIR` "
                                    "(or tests/mock_release.py for mock data)")
        self.db = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.ancestors = lru_cache(maxsize=4096)(self._ancestors)  # codes are frozen, hashable
        self._isa_up = lru_cache(maxsize=65536)(self._isa_ancestors)

    def _q(self, sql: str, *args) -> list[tuple]:
        return self.db.execute(sql, args).fetchall()

    def lookup(self, code: Code) -> Code:
        if code.system == "SCT":
            row = self._q("SELECT fsn FROM sct_concept WHERE id = ?", code.code)
            if not row:
                raise UnknownCode(f"unknown SNOMED code {code}")
            return Code("SCT", code.code, _name(row[0][0]))
        if code.system == "LOINC":
            row = self._q("SELECT long_common_name FROM loinc_term WHERE loinc = ?", code.code)
            if not row:
                raise UnknownCode(f"unknown LOINC code {code}")
            return Code("LOINC", code.code, row[0][0])
        return code

    def _ancestors(self, code: Code) -> list[Code]:
        if code.system != "SCT":
            return [code]
        self.lookup(code)
        out, frontier = [code.code], [code.code]
        while frontier:  # BFS: nearest first
            nxt = []
            for c in frontier:
                ups = self._q("SELECT parent FROM sct_isa WHERE child = ?", c) + \
                    self._q("SELECT value FROM sct_attr WHERE concept = ? AND type = ?", c, PART_OF)
                for (p,) in ups:
                    if p not in out:
                        out.append(p)
                        nxt.append(p)
            frontier = nxt
        return [self.lookup(Code.sct(c)) for c in out]

    def finding_groups(self, finding: Code) -> list[tuple[Code | None, Code | None]]:
        self.lookup(finding)
        groups: dict[int, dict[str, str]] = {}
        for g, t, v in self._q("SELECT grp, type, value FROM sct_attr WHERE concept = ? ORDER BY rowid", finding.code):
            groups.setdefault(g, {})[t] = v
        out = []
        for g in sorted(groups):
            s, m = groups[g].get(FINDING_SITE), groups[g].get(ASSOCIATED_MORPHOLOGY)
            if s or m:
                out.append((s and self.lookup(Code.sct(s)), m and self.lookup(Code.sct(m))))
        return out

    def _lateral(self, cid: str) -> str | None:
        rows = self._q("SELECT value FROM sct_attr WHERE concept = ? AND type = ? ORDER BY rowid", cid, LATERALITY)
        return next((v for (v,) in rows if v in SIDES), None)

    def split_laterality(self, site: Code) -> tuple[Code, Code | None]:
        self.lookup(site)
        lat = self._lateral(site.code)
        if not lat:
            return site, None
        for (p,) in self._q("SELECT parent FROM sct_isa WHERE child = ? ORDER BY rowid", site.code):
            if not self._lateral(p):
                return self.lookup(Code.sct(p)), self.lookup(Code.sct(lat))
        return site, None

    def procedure_parts(self, loinc: Code) -> ProcedureParts:
        self.lookup(loinc)
        parts = {t: (num, name) for t, num, name in self._q(
            "SELECT part_type, part_number, part_name FROM loinc_part WHERE loinc = ? ORDER BY rowid", loinc.code)}

        def sct(part_type: str) -> Code | None:
            return self.part_sct(part_type, *parts[part_type]) if part_type in parts else None

        mod = parts.get("Rad.Modality.Modality Type")
        return ProcedureParts(
            modality=Code("DCM", RAD_MODALITY[mod[1]]) if mod and mod[1] in RAD_MODALITY else None,
            body_site=sct("Rad.Anatomic Location.Imaging Focus") or sct("Rad.Anatomic Location.Region Imaged"),
            laterality=sct("Rad.Anatomic Location.Laterality"),
        )

    def part_sct(self, part_type: str, num: str, name: str) -> Code | None:
        """Rad anatomic part -> SNOMED. PartRelatedCodeMapping first; real releases map these parts to RadLex only,
        so then by name: sides by table, sites by exact description under Body structure."""
        row = self._q("SELECT sct FROM loinc_part_sct WHERE part_number = ?", num)
        if row:
            return self.lookup(Code.sct(row[0][0]))
        if part_type.endswith("Laterality"):
            side = RAD_SIDE.get(name.lower())
            return side and self.lookup(Code.sct(side))
        for term in part_terms(name):
            if hits := self.find(term, BODY_STRUCTURE):
                return hits[0]
        raise UnknownCode(f"LOINC part {num} {name} has no SNOMED mapping")

    def _isa_ancestors(self, cid: str) -> set[str]:
        return {r[0] for r in self._q(UP, cid)}

    def find(self, term: str, root: str) -> list[Code]:
        rows = self._q("SELECT DISTINCT concept FROM sct_description WHERE term_lower = ? ORDER BY concept", term.lower())
        return [self.lookup(Code.sct(c)) for (c,) in rows if root in self._isa_up(c)]

    def search(self, text: str, root: str, limit: int = 10) -> list[Code]:
        # match first (one scan), then check the root upward: cheap even for huge roots like Clinical finding
        rows = self._q("SELECT concept, MIN(length(term)) m FROM sct_description WHERE instr(term_lower, ?) > 0"
                       " GROUP BY concept ORDER BY m, concept", text.lower())
        out = []
        for c, _ in rows:
            if root in self._isa_up(c):
                out.append(self.lookup(Code.sct(c)))
                if len(out) == limit:
                    break
        return out
