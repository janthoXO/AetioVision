import numpy as np
import pytest
from PIL import Image

from aetiovision.domain.codes import LEFT, RIGHT, BILATERAL, Code
from aetiovision.domain.errors import MaskingFailed
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.template import Template
from aetiovision.masking.grounded import choose_boxes, parse_boxes
from aetiovision.masking.predefined import PredefinedMasker, feather, qa

SITE = Code.sct("41111004", "Shaft of femur")


def template(masks=None, shape=(100, 80)):
    return Template("t", MedicalImage(np.zeros(shape, np.float32)), Code("DCM", "DX"), SITE, None, None, "s", "l", masks or {})


def test_parse_boxes():
    assert parse_boxes('```json\n[{"bbox_2d": [100, 200, 500, 800], "label": "x"}]\n```', 1000, 500) == [[100, 100, 500, 400]]
    assert parse_boxes('{"bbox_2d": [0, 0, 1000, 1000]}', 200, 100) == [[0, 0, 200, 100]]
    assert parse_boxes('sure: [{"bbox_2d": [-5, 0, 2000, 500]}]', 100, 100) == [[0, 0, 100, 50]]
    assert parse_boxes('[{"bbox_2d": [10, 10, 10, 500]}]', 100, 100) == []  # degenerate
    assert parse_boxes("[]", 100, 100) == [] and parse_boxes("no idea", 100, 100) == []


def test_choose_boxes():
    a, b = [0, 0, 10, 10], [50, 0, 100, 40]
    assert choose_boxes([b, a], Code.sct(RIGHT), 100) == [a]  # patient right = image left
    assert choose_boxes([a, b], Code.sct(LEFT), 100) == [b]
    assert choose_boxes([a, b], Code.sct(BILATERAL), 100) == [a, b]
    assert choose_boxes([a, b], None, 100) == [b]
    assert choose_boxes([a], Code.sct(LEFT), 100) == [a]


def test_qa():
    for bad in (np.zeros((100, 100), bool), np.ones((100, 100), bool)):
        with pytest.raises(MaskingFailed):
            qa(bad)
    m = np.zeros((100, 100), bool)
    m[10:40, 10:40] = True
    m[80:82, 80:82] = True  # tiny component dropped
    out = qa(m)
    assert out[20, 20] and not out[80, 80]


def test_feather():
    b = np.zeros((50, 60), bool)
    b[20:30, 20:30] = True
    a = feather(b, 2.0)
    assert a.shape == b.shape and a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1 and 0 < a[19, 25] < 1
    thin = np.zeros((100, 100), bool)
    thin[:, 50:52] = True  # 2 px wide, sigma 10: plain blur would peak far below 0.5
    assert feather(thin, 10.0)[:, 50:52].min() == 1


def test_predefined(tmp_path):
    png = tmp_path / "m.png"
    m = np.zeros((50, 40), np.uint8)  # resized (nearest) to the template
    m[10:40, 10:30] = 255
    Image.fromarray(m).save(png)
    r = PredefinedMasker().mask(template({str(SITE): png}), SITE, None)
    assert r.alpha.shape == (100, 80) and r.alpha.dtype == np.float32 and r.alpha[50, 40] > 0.99 and r.alpha[0, 0] < 0.01
    assert r.source == f"predefined:{png}"


class Stub:
    def open_vocabulary(self):
        return True

    def mask(self, template, site, laterality):
        a = np.zeros((100, 80), np.float32)
        a[20:60, 20:50] = 1
        return Mask(a, "stub")


def test_fallback_and_failures():
    pm = PredefinedMasker(Stub())
    assert pm.open_vocabulary() and pm.mask(template(), SITE, None).source == "stub"
    with pytest.raises(MaskingFailed):
        PredefinedMasker().mask(template(), SITE, None)
    assert not PredefinedMasker().open_vocabulary()

    class Empty(Stub):
        def mask(self, *a):
            return Mask(np.zeros((100, 80), np.float32), "e")

    with pytest.raises(MaskingFailed):
        PredefinedMasker(Empty()).mask(template(), SITE, None)
