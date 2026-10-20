import json
from pathlib import Path

import numpy as np
from PIL import Image

from aetiovision.catalog.sqlite import MAX_CANDIDATES, SqliteCatalog
from aetiovision.domain.codes import Code, Triple
from aetiovision.domain.image import Mask, MedicalImage
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.result import Result
from aetiovision.utils.images import load_gray, preview_png, save_result
from fakes import blank_template

LOWER, HAND, FEMUR, LEFT = "SCT:61685007", "SCT:85562004", "SCT:71341001", "SCT:7771000"


def req(site="71341001", lat=None, seed=1, modality="DX"):
    t = Triple(Code("DCM", modality), Code.sct(site), Code.sct("72704001"))
    return ResolvedRequest(t, t, lat and Code.parse(lat), "", seed)


def png(path: Path, size=(200, 100), mode="L", value=128) -> Path:
    Image.new(mode, size, value).save(path)
    return path


def make(tmp_path):
    return SqliteCatalog(tmp_path / "c.db", tmp_path / "data")


def test_add_and_load(tmp_path):
    cat = make(tmp_path)
    img, mask = png(tmp_path / "a.png", (2000, 1000)), png(tmp_path / "m.png", (2000, 1000), value=255)
    assert cat.add(img, "DCM:DX", LOWER, source="s", license="l", masks={FEMUR: mask}) == "a"
    (t,) = cat.candidates(req(), set())
    assert t.id == "a" and t.image.meta == {"template": "a"} and t.image.pixels.shape == (512, 1024)
    assert t.image.pixels.dtype == np.float32 and abs(t.image.pixels.mean() - 128 / 255) < 0.01
    assert list(t.masks) == [FEMUR] and t.masks[FEMUR].exists()
    assert Image.open(t.masks[FEMUR]).size == (1024, 512)
    assert cat.coverage() == {(Code("DCM", "DX"), Code.parse(FEMUR))}
    assert cat.regions() == {(Code("DCM", "DX"), Code.parse(LOWER))}


def test_add_16bit_and_replace(tmp_path):
    cat = make(tmp_path)
    a = (np.arange(100 * 100).reshape(100, 100) * 6).astype(np.uint16)
    p = tmp_path / "x.png"
    Image.fromarray(a).save(p)
    cat.add(p, "DCM:DX", LOWER, source="s", license="l", id="x", masks={FEMUR: png(tmp_path / "m.png")})
    cat.add(p, "DCM:DX", LOWER, source="s", license="l", id="x")  # replace drops masks
    assert cat.coverage() == set()
    px = load_gray(tmp_path / "data/images/x.png")
    assert px.min() == 0 and px.max() == 1


def test_candidate_groups_and_cap(tmp_path):
    cat = make(tmp_path)
    for i in range(4):
        cat.add(png(tmp_path / f"m{i}.png"), "DCM:DX", LOWER, source="s", license="l", masks={FEMUR: png(tmp_path / "mk.png")})
    for i in range(4):
        cat.add(png(tmp_path / f"r{i}.png"), "DCM:DX", LOWER, source="s", license="l")
    cat.add(png(tmp_path / "h.png"), "DCM:DX", HAND, source="s", license="l")
    cat.add(png(tmp_path / "ct.png"), "DCM:CT", LOWER, source="s", license="l")
    no_fallback = cat.candidates(req(), set())
    assert sorted(t.id for t in no_fallback) == ["m0", "m1", "m2", "m3"]
    both = cat.candidates(req(), {Code.parse(LOWER)})
    assert len(both) == MAX_CANDIDATES == 5
    assert all(t.id.startswith("m") for t in both[:4]) and both[4].id.startswith("r")
    assert not any(t.id in ("h", "ct") for t in both)
    ids = lambda s: [t.id for t in cat.candidates(req(seed=s), {Code.parse(LOWER)})]
    assert ids(3) == ids(3)
    assert any(ids(s) != ids(0) for s in range(1, 10))


def test_laterality_filter(tmp_path):
    cat = make(tmp_path)
    cat.add(png(tmp_path / "any.png"), "DCM:DX", LOWER, source="s", license="l")
    cat.add(png(tmp_path / "left.png"), "DCM:DX", LOWER, laterality=LEFT, source="s", license="l")
    reg = {Code.parse(LOWER)}
    assert {t.id for t in cat.candidates(req(), reg)} == {"any"}
    assert {t.id for t in cat.candidates(req(lat=LEFT), reg)} == {"any", "left"}
    assert {t.id for t in cat.candidates(req(lat="SCT:24028007"), reg)} == {"any"}


def test_save_result(tmp_path):
    t = blank_template("tpl")
    r = req(seed=7)
    res = Result(r, t, Mask(np.ones((64, 64), np.float32), "predefined:m.png"), MedicalImage(np.zeros((64, 64), np.float32)), "fake")
    paths = save_result(res, tmp_path / "out")
    assert set(paths) == {"image", "mask", "template", "json"}
    assert all(Path(p).is_absolute() and Path(p).exists() for p in paths.values())
    assert Path(paths["image"]).name.endswith("_tpl_7.png")
    d = json.loads(Path(paths["json"]).read_text())
    assert d["requested"]["modality"] == "DCM:DX" and d["template"]["id"] == "tpl" and d["mask"]["source"] == "predefined:m.png"
    assert preview_png(res.image.pixels, 32)[:4] == b"\x89PNG"
