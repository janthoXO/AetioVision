import numpy as np
import pytest

from aetiovision.domain.codes import LEFT, Code, Triple
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.template import Template
from aetiovision.inpainting.diffusion import DiffusionInpainter, build_prompt, crop_box, paste
from aetiovision.inpainting.procedural import BONE, FRACTURE, NEOPLASM, ProceduralInpainter

DX = Code("DCM", "DX")


def req(morph, abstract=FRACTURE, instr="", seed=1, site=BONE):
    sup = Triple(DX, Code.sct(site), Code.sct(abstract))
    return ResolvedRequest(Triple(DX, Code.sct(site, "Femur"), Code.sct(morph, "Fracture, transverse")), sup, Code.sct(LEFT, "Left"), instr, seed)


def setup():
    img = (0.5 + 0.1 * np.random.default_rng(0).random((200, 160))).astype(np.float32)
    yy, xx = np.mgrid[:200, :160]
    a = np.clip((1 - (((yy - 100) / 70) ** 2 + ((xx - 80) / 20) ** 2) ** 0.5) * 4, 0, 1).astype(np.float32)
    return Template("t", MedicalImage(img), DX, Code.sct(BONE), None, None, "", ""), Mask(a)


@pytest.mark.parametrize("m", ["70682000", "7905002", "13321001", FRACTURE])
def test_fracture(m):
    t, mask = setup()
    p = ProceduralInpainter()
    r = req(m)
    a, b = p.inpaint(t, mask, r), p.inpaint(t, mask, r)
    assert np.array_equal(a.pixels, b.pixels) and a.pixels.dtype == np.float32
    assert 0 <= a.pixels.min() and a.pixels.max() <= 1 and a.meta["inpainter"] == "procedural"
    assert not np.allclose(a.pixels, t.image.pixels)
    assert np.array_equal(a.pixels[mask.alpha == 0], t.image.pixels[mask.alpha == 0])
    assert not np.array_equal(a.pixels, p.inpaint(t, mask, req(m, seed=2)).pixels)


def test_displaced_and_neoplasm():
    t, mask = setup()
    p = ProceduralInpainter()
    d = p.inpaint(t, mask, req("70682000", instr="Displaced"))
    assert d.meta["displaced"] and not np.allclose(d.pixels, t.image.pixels)
    n = p.inpaint(t, mask, req(NEOPLASM, NEOPLASM, site="123037004"))
    assert n.pixels.mean() > t.image.pixels.mean() and np.array_equal(n.pixels[mask.alpha == 0], t.image.pixels[mask.alpha == 0])


def test_capabilities():
    assert len(ProceduralInpainter().capabilities()) == 2
    assert len(DiffusionInpainter().capabilities()) == 6


def test_prompt():
    p = build_prompt(req("70682000", instr="displaced shaft"))
    assert "femur" in p and "left" in p and "displaced shaft" in p and "x-ray" in p


def test_crop_paste():
    a = np.zeros((300, 200), np.float32)
    a[100:140, 50:70] = 1
    box = y0, y1, x0, x1 = crop_box(a, 0.3)
    assert y1 - y0 == x1 - x0 and y0 >= 0 and y1 <= 300 and x0 >= 0 and x1 <= 200
    assert y0 <= 100 and y1 >= 140 and x0 <= 50 and x1 >= 70
    img = np.random.default_rng(0).random((300, 200)).astype(np.float32)
    out = paste(img, img[y0:y1, x0:x1], box)
    assert np.array_equal(out, img) and out is not img
    out = paste(img, np.zeros((64, 64), np.float32), box)
    assert np.array_equal(out[:y0], img[:y0]) and out[y0:y1, x0:x1].max() == 0
