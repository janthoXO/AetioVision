import os

import pytest
from fakes import FakeInpainter, FakeTerminology

from aetiovision.domain.codes import Code, Triple
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.resolution import resolve
from aetiovision.pipeline.supported import SupportedSet
from aetiovision.text_extraction.rules import RulesExtractor, map_term

DX, CT = Code("DCM", "DX"), Code("DCM", "CT")
LOWER, UPPER = Code.sct("61685007"), Code.sct("53120007")
BONE, FRACTURE = Code.sct("272673000"), Code.sct("72704001")
T = FakeTerminology()
SUP = SupportedSet(T, set(), {(DX, LOWER), (DX, UPPER)}, [FakeInpainter(caps={Triple(DX, BONE, FRACTURE)})])
X = RulesExtractor(T)


def codes(r: CodeRequest):
    return r.modality, r.body_site and r.body_site.code, r.morphology and r.morphology.code, r.laterality and r.laterality.code


def test_femoral_shaft():
    text = "AP x-ray of the left femoral shaft with a displaced transverse fracture"
    r = X.extract(text, SUP)
    assert codes(r) == (DX, "41111004", "70682000", "7771000") and r.instructions == text and r.seed is None
    assert resolve(r, T, SUP).requested.body_site.code == "41111004"


def test_metacarpal_german_laterality():
    r = X.extract("fracture of the fifth metacarpal, right hand, radiograph", SUP)
    assert codes(r) == (DX, "50939004", "72704001", "24028007")
    assert resolve(r, T, SUP).supported.body_site.code == "50939004"
    assert X.extract("Röntgen Oberschenkel links", SUP).laterality.code == "7771000"


def test_ct_head_and_nothing():
    r = X.extract("CT of the head", SupportedSet(T, set(), {(CT, Code.sct("69536005"))}, []))
    assert codes(r) == (CT, "69536005", None, None)
    assert codes(X.extract("please make something nice", SUP)) == (None, None, None, None)


def test_map_term():
    assert map_term(T, SUP, "Shaft of femur", "123037004", DX).code == "41111004"
    assert map_term(T, SUP, "transverse fracture", "49755003").code == "70682000"
    assert map_term(T, SUP, "fracture", "123037004") is None  # morphologies are not body sites


@pytest.mark.skipif(os.environ.get("AETIO_TEST_LLM") != "1", reason="needs a local Ollama")
def test_llm_ollama():
    from aetiovision.text_extraction.llm import LlmExtractor

    r = LlmExtractor(T, "ollama", os.environ.get("EXTRACTION_MODEL", "qwen3:14b"), None).extract(
        "Röntgen linker Oberschenkel, Querfraktur im Schaft", SUP
    )
    assert codes(r)[0] == DX and r.laterality.code == "7771000"
