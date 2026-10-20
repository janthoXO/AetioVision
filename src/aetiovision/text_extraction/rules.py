"""Offline rule-based extractor (D18 fallback): regexes for modality/laterality, n-gram lookup for the rest."""

from __future__ import annotations

import re

from aetiovision.domain.codes import (
    BILATERAL,
    BODY_STRUCTURE,
    CLINICAL_FINDING,
    LEFT,
    MORPHOLOGY,
    RIGHT,
    Code,
)
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.ports import Terminology
from aetiovision.pipeline.supported import SupportedSet

MODALITY = {
    "DX": r"x-?\s?rays?|radiographs?|radiography|r[öo]e?ntgen\w*|plain films?",
    "CT": r"ct|computed tomography|computertomographie",
    "MR": r"mri?|magnetic resonance",
}
LATERALITY = {
    LEFT: r"left|links\w*",
    RIGHT: r"right|rechts\w*",
    BILATERAL: r"bilateral|beidseits|beidseitig",
}
STOPWORDS = {"a", "an", "the", "of", "with", "and", "in", "on", "at", "to", "for", "or", "ap", "pa", "lateral", "view"}
ADJECTIVES = {  # adjective -> noun, for "femoral shaft" -> "shaft of femur"
    "femoral": "femur", "tibial": "tibia", "humeral": "humerus", "radial": "radius",
    "ulnar": "ulna", "metacarpal": "metacarpal", "pelvic": "pelvis",
}


def first_match(patterns: dict[str, str], text: str) -> str | None:
    """Key of the pattern matching earliest in text."""
    hits = [(m.start(), k) for k, p in patterns.items() if (m := re.search(rf"\b(?:{p})\b", text, re.I))]
    return min(hits)[1] if hits else None


def rank(supported: SupportedSet, cands: list[Code], root: str, modality: Code | None) -> Code | None:
    """First candidate (morphologies excluded from body sites), preferring supported ones."""
    if root == BODY_STRUCTURE:
        cands = [c for c in cands if Code.sct(MORPHOLOGY) not in supported.ancestors(c)]
    for c in cands:
        if root == BODY_STRUCTURE and supported.matching(modality, c, None):
            return c
        if root == MORPHOLOGY and supported.matching(modality, None, c):
            return c
    return cands[0] if cands else None


def map_term(terminology: Terminology, supported: SupportedSet, term: str, root: str, modality: Code | None = None) -> Code | None:
    """Text term -> SNOMED code: exact synonym match first, then substring search, prefer supported."""
    term = term.strip().lower()
    if not term:
        return None
    return rank(supported, terminology.find(term, root), root, modality) or rank(
        supported, terminology.search(term, root), root, modality
    )


def variants(gram: str) -> list[str]:
    out = [gram]
    norm = " ".join(ADJECTIVES.get(w, w) for w in gram.split())
    if (m := re.fullmatch(r"(\w+) shaft", norm)):
        norm = f"shaft of {m[1]}"
    out += [v for v in (norm, gram + " bone", norm + " bone") if v not in out]
    return out


class RulesExtractor:
    def __init__(self, terminology: Terminology):
        self.t = terminology

    def extract(self, text: str, supported: SupportedSet) -> CodeRequest:
        low = text.lower()
        mod = first_match(MODALITY, low)
        lat = first_match(LATERALITY, low)
        modality = Code("DCM", mod) if mod else None
        words = re.findall(r"[^\W_]+(?:-[^\W_]+)*", low)
        grams = [(n, " ".join(words[i : i + n])) for n in range(6, 0, -1) for i in range(len(words) - n + 1)]
        grams = [(n, g) for n, g in grams if not set(g.split()) <= STOPWORDS]
        site = self._best(grams, BODY_STRUCTURE, modality, supported)
        morph = self._best(grams, MORPHOLOGY, modality, supported)
        finding = self._best([x for x in grams if x[0] >= 3], CLINICAL_FINDING, modality, supported)
        return CodeRequest(
            modality=modality,
            body_site=site,
            morphology=morph,
            laterality=lat and Code.sct(lat),
            finding=finding,
            instructions=text,
        )

    def _best(self, grams, root, modality, supported) -> Code | None:
        best, key = None, (-1, -1)
        for n, g in grams:  # longest first, so on ties the earlier (longer) gram stays
            for v in variants(g):
                c = rank(supported, self.t.find(v, root), root, modality)
                if not c:
                    continue
                ok = root == CLINICAL_FINDING or bool(
                    supported.matching(modality, c if root == BODY_STRUCTURE else None, c if root == MORPHOLOGY else None)
                )
                if (ok, n) > key:
                    best, key = c, (ok, n)
        return best
