"""Codes -> ResolvedRequest. Rules 1-6 of design section 2, pure over Terminology + SupportedSet."""

from __future__ import annotations

import secrets

from aetiovision.domain.codes import (
    BILATERAL,
    BODY_STRUCTURE,
    CLINICAL_FINDING,
    LEFT,
    MORPHOLOGY,
    RIGHT,
    Code,
    Triple,
)
from aetiovision.domain.errors import AmbiguousCode, CodeConflict, InvalidCode, UnsupportedRequest
from aetiovision.domain.request import CodeRequest, ResolvedRequest
from aetiovision.pipeline.ports import Terminology
from aetiovision.pipeline.supported import SupportedSet

MODALITY_NAMES = {"DX": "X-ray radiograph", "CT": "CT", "MR": "MRI"}
SLOTS = ("modality", "body_site", "morphology", "laterality")


def resolve(req: CodeRequest, term: Terminology, sup: SupportedSet) -> ResolvedRequest:
    proposals: dict[str, list[tuple[Code, str]]] = {s: [] for s in SLOTS}

    def check(code: Code, root: str, what: str) -> Code:
        code = term.lookup(code)
        anc = sup.ancestors(code)
        # morphologic abnormality IS-A body structure in SNOMED: a body site must not be one
        if code.system != "SCT" or Code.sct(root) not in anc or (root == BODY_STRUCTURE and Code.sct(MORPHOLOGY) in anc):
            raise InvalidCode(f"{code.label()} is not a {what}")
        return code

    # 1. collect
    if req.modality:
        proposals["modality"].append((req.modality, "modality"))
    if req.body_site:
        proposals["body_site"].append((check(req.body_site, BODY_STRUCTURE, "body structure"), "body_site"))
    if req.morphology:
        proposals["morphology"].append((check(req.morphology, MORPHOLOGY, "morphologic abnormality"), "morphology"))
    if req.laterality:
        if req.laterality.system != "SCT" or req.laterality.code not in (LEFT, RIGHT, BILATERAL):
            raise InvalidCode(f"laterality must be SCT:{LEFT}, SCT:{RIGHT} or SCT:{BILATERAL}, got {req.laterality}")
        proposals["laterality"].append((term.lookup(req.laterality), "laterality"))
    if req.procedure:
        if req.procedure.system != "LOINC":
            raise InvalidCode(f"procedure must be a LOINC code, got {req.procedure}")
        parts = term.procedure_parts(req.procedure)
        src = f"procedure {req.procedure}"
        for slot, code in (("modality", parts.modality), ("body_site", parts.body_site), ("laterality", parts.laterality)):
            if code:
                proposals[slot].append((code, src))
    if req.finding:
        finding = check(req.finding, CLINICAL_FINDING, "clinical finding")
        src = f"finding {finding}"
        site, morph = _pick_group(finding, term.finding_groups(finding), req, sup)
        if site:
            proposals["body_site"].append((site, src))
        if morph:
            proposals["morphology"].append((morph, src))

    # 2. split laterality off lateralized sites
    for i, (site, src) in enumerate(list(proposals["body_site"])):
        base, lat = term.split_laterality(site)
        if lat:
            proposals["body_site"][i] = (term.lookup(base), src)
            proposals["laterality"].append((term.lookup(lat), src))

    # 3. merge per slot
    merged: dict[str, Code | None] = {}
    sources: dict[str, str] = {}
    for slot, props in proposals.items():
        merged[slot] = None
        for code, src in props:
            cur = merged[slot]
            if cur is None or code in sup.ancestors(cur):
                if cur is None:
                    merged[slot], sources[slot] = code, src
            elif cur in sup.ancestors(code):
                merged[slot], sources[slot] = code, src
            else:
                raise CodeConflict(f"{slot}: {cur.label()} ({sources[slot]}) vs {code.label()} ({src})")

    # 5. fill gaps (D19)
    assumptions: list[str] = []
    m, site, morph = merged["modality"], merged["body_site"], merged["morphology"]
    rows = sup.matching(m, site, morph)
    if not rows:
        given = Triple(m or Code("DCM", "?"), site or Code.sct("?"), morph or Code.sct("?"))
        raise UnsupportedRequest(f"nothing supported for {given.label()}", sup.listing())
    if m is None:
        options = {r.modality for r in rows}
        if len(options) != 1:
            raise UnsupportedRequest("modality missing and not unique", sorted(map(str, options)))
        m = options.pop()
        sources["modality"] = "inferred"
        assumptions.append(f"modality {m}: only supported modality for the other codes")
    if site is None:
        options = {(r.site, r.open) for r in rows if r.modality == m}
        if len(options) != 1 or next(iter(options))[1]:
            raise UnsupportedRequest(
                "body_site missing and not unique",
                sorted(("any site within " if o else "") + term.lookup(s).label() for s, o in options),
            )
        site = term.lookup(options.pop()[0])
        sources["body_site"] = "inferred"
        assumptions.append(f"body_site {site.label()}: only supported site for the other codes")
    if morph is None:
        options = {r.morphology for r in sup.matching(m, site, None)}
        if len(options) != 1:
            raise UnsupportedRequest("morphology missing and not unique", sorted(term.lookup(o).label() for o in options))
        morph = term.lookup(options.pop())
        sources["morphology"] = "inferred"
        assumptions.append(f"morphology {morph.label()}: only supported morphology for the other codes")

    # 6. roll up
    m = Code("DCM", m.code, MODALITY_NAMES.get(m.code, m.code))
    requested = Triple(m, site, morph)
    rolled = sup.rollup(requested)
    if rolled is None:
        raise UnsupportedRequest(f"not supported: {requested.label()}", sup.listing())
    supported = Triple(m, term.lookup(rolled.body_site), term.lookup(rolled.morphology))
    return ResolvedRequest(
        requested=requested,
        supported=supported,
        laterality=merged["laterality"],
        instructions=req.instructions,
        seed=req.seed if req.seed is not None else secrets.randbelow(2**31),
        sources=sources,
        assumptions=assumptions,
    )


def _pick_group(finding, groups, req: CodeRequest, sup: SupportedSet) -> tuple[Code | None, Code | None]:
    """Rule 4: keep role groups that roll up to the supported set."""
    if not groups:
        return None, None
    if len(groups) == 1:
        return groups[0]
    ok = [g for g in groups if sup.matching(req.modality, g[0] or req.body_site, g[1] or req.morphology)]
    if len(ok) == 1:
        return ok[0]
    shown = ok or groups
    raise AmbiguousCode(
        f"finding {finding.label()} has {len(shown)} role groups: "
        + "; ".join(f"site {s.label() if s else '-'}, morphology {m.label() if m else '-'}" for s, m in shown)
    )
