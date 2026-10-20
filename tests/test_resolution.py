import pytest
from fakes import FakeInpainter, FakeTerminology

from aetiovision.domain.codes import Code, Triple
from aetiovision.domain.errors import CodeConflict, InvalidCode, UnknownCode, UnsupportedRequest
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.resolution import resolve
from aetiovision.pipeline.supported import SupportedSet

DX, CT = Code("DCM", "DX"), Code("DCM", "CT")
SHAFT, FEMUR, LOWER_LIMB, UPPER_LIMB = (Code.sct(c) for c in ("41111004", "71341001", "61685007", "53120007"))
FRACTURE, TRANSVERSE, NEOPLASM = (Code.sct(c) for c in ("72704001", "70682000", "108369006"))
LEFT = Code.sct("7771000")
BONE = Code.sct("272673000")

T = FakeTerminology()


def sup(masks=((DX, SHAFT),), regions=(), caps=((DX, BONE, FRACTURE),)):
    return SupportedSet(T, set(masks), set(regions), [FakeInpainter(caps={Triple(*c) for c in caps})])


def test_explicit_triple():
    r = resolve(CodeRequest(DX, SHAFT, TRANSVERSE, LEFT, instructions="displaced", seed=1), T, sup())
    assert r.requested.morphology == TRANSVERSE and r.supported.morphology == FRACTURE
    assert r.supported.body_site == SHAFT and r.laterality == LEFT and r.seed == 1
    assert r.supported.body_site.display == "Bone structure of shaft of femur"


def test_finding_plus_modality():
    r = resolve(CodeRequest(modality=DX, finding=Code.sct("54441004")), T, sup())  # Fracture of shaft of femur
    assert r.supported == Triple(DX, SHAFT, FRACTURE)
    assert r.sources["body_site"].startswith("finding")


def test_finding_only_infers_modality():
    r = resolve(CodeRequest(finding=Code.sct("54441004")), T, sup())
    assert r.supported.modality == DX and r.assumptions


def test_lateralized_finding_splits():
    r = resolve(CodeRequest(modality=DX, body_site=Code.sct("758654003"), morphology=FRACTURE), T, sup())
    assert r.requested.body_site == SHAFT and r.laterality == LEFT


def test_loinc_procedure_plus_finding():
    # XR Femur - left: DX + femur + left; finding gives shaft (more specific) + fracture
    r = resolve(CodeRequest(procedure=Code("LOINC", "36654-2"), finding=Code.sct("54441004")), T, sup())
    assert r.requested.body_site == SHAFT and r.laterality == LEFT and r.supported.modality == DX


def test_conflict():
    with pytest.raises(CodeConflict):
        resolve(CodeRequest(DX, SHAFT, FRACTURE, procedure=Code("LOINC", "24725-4")), T, sup())  # CT Head


def test_wrong_hierarchy_and_unknown():
    with pytest.raises(InvalidCode):
        resolve(CodeRequest(DX, FRACTURE, FRACTURE), T, sup())
    with pytest.raises(UnknownCode):
        resolve(CodeRequest(DX, Code.sct("999999"), FRACTURE), T, sup())


def test_unsupported_lists_candidates():
    with pytest.raises(UnsupportedRequest) as e:
        resolve(CodeRequest(DX, SHAFT, NEOPLASM), T, sup())
    assert e.value.candidates


def test_rollup_to_ancestor_mask():
    r = resolve(CodeRequest(DX, Code.sct("33003004"), FRACTURE), T, sup())  # proximal shaft -> shaft mask
    assert r.supported.body_site == SHAFT and r.requested.body_site.code == "33003004"


def test_open_region_keeps_site():
    s = sup(masks=(), regions=((DX, UPPER_LIMB), (DX, LOWER_LIMB)))
    r = resolve(CodeRequest(DX, Code.sct("50939004"), FRACTURE), T, s)  # fifth metacarpal
    assert r.supported.body_site.code == "50939004"
    with pytest.raises(UnsupportedRequest):  # site missing, regions are open: not unique
        resolve(CodeRequest(DX, morphology=FRACTURE), T, s)


def test_morphology_inferred():
    r = resolve(CodeRequest(DX, SHAFT), T, sup())
    assert r.supported.morphology == FRACTURE and any("morphology" in a for a in r.assumptions)
