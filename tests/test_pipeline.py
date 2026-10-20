import numpy as np
import pytest
from fakes import FakeCatalog, FakeExtractor, FakeInpainter, FakeMasker, FakeTerminology, blank_template

from aetiovision.domain.codes import Code
from aetiovision.domain.errors import MaskingFailed, NoTemplate
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.pipeline import Pipeline

REQ = CodeRequest(Code("DCM", "DX"), Code.sct("41111004"), Code.sct("72704001"), seed=3)


def make(templates=None, masker=None, extractor=None):
    cat = FakeCatalog(templates) if templates is not None else FakeCatalog()
    return Pipeline(FakeTerminology(), extractor or FakeExtractor(REQ), cat, masker or FakeMasker(), [FakeInpainter()])


def test_end_to_end_and_no_leakage():
    p = make()
    r = p.from_codes(REQ)
    a, src, out = r.mask.alpha, r.template.image.pixels, r.image.pixels
    assert np.array_equal(out[a == 0], src[a == 0])  # leakage invariant
    assert np.allclose(out[a == 1], 1.0)
    assert r.inpainter == "fake" and out.dtype == np.float32


def test_from_text_uses_extractor():
    assert make().from_text("anything", seed=5).resolved.seed == 5


def test_region_template_retry():
    ts = [blank_template(f"r{i}") for i in range(3)]  # region only, no predefined masks
    masker = FakeMasker(fail_ids=("r0", "r1"))
    r = make(ts, masker).from_codes(REQ)
    assert r.template.id == "r2" or r.template.id not in masker.fail_ids


def test_masking_fails_on_all():
    ts = [blank_template(f"r{i}") for i in range(2)]
    with pytest.raises(MaskingFailed):
        make(ts, FakeMasker(fail_ids=("r0", "r1"))).from_codes(REQ)


def test_no_template_without_open_masker():
    # closed masker + region-only templates: nothing in the supported set
    with pytest.raises(Exception) as e:
        make([blank_template("r0")], FakeMasker(open=False)).from_codes(REQ)
    assert type(e.value).__name__ in ("UnsupportedRequest", "NoTemplate")


def test_no_template_for_laterality():
    ts = [blank_template("r0", laterality="24028007")]  # right only
    with pytest.raises(NoTemplate):
        make(ts).from_codes(CodeRequest(Code("DCM", "DX"), Code.sct("41111004"), Code.sct("72704001"), Code.sct("7771000")))
