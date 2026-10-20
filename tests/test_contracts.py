"""SqliteTerminology (imported from mock release files) must answer like FakeTerminology."""

import sqlite3

import pytest
from fakes import FakeTerminology
from mock_release import GHOST, MOCK, ROOT, SHAFT, write_mock_release
from test_resolution import DX, FRACTURE, LEFT, TRANSVERSE, sup
from test_resolution import SHAFT as SHAFT_CODE

from aetiovision.domain.codes import BODY_STRUCTURE, Code
from aetiovision.domain.errors import UnknownCode
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.resolution import resolve
from aetiovision.pipeline.supported import SupportedSet
from aetiovision.terminology.importer import import_terminology
from aetiovision.terminology.sqlite import SqliteTerminology
from fetch_mock_terminology import SEED_SCT

FAKE = FakeTerminology()
CODES = [Code.sct(c) for c in sorted(set(SEED_SCT) | {"138875005", BODY_STRUCTURE, "49755003", "404684003"})]
LOINCS = [Code("LOINC", c) for c in MOCK["loinc"]]


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("release")
    snomed, loinc = write_mock_release(tmp)
    import_terminology(snomed, loinc, tmp / "terminology.db")
    assert not (tmp / "terminology.tmp").exists()
    return SqliteTerminology(tmp / "terminology.db"), tmp / "terminology.db"


@pytest.fixture(scope="module")
def sq(db):
    return db[0]


def test_lookup(sq):
    for c in CODES + LOINCS + [Code("DCM", "DX")]:
        r = sq.lookup(c)
        assert (r, r.display) == (FAKE.lookup(c), FAKE.lookup(c).display)
    for bad in (Code.sct("999999"), Code("LOINC", "0-0"), Code.sct(GHOST)):  # GHOST is inactive
        with pytest.raises(UnknownCode):
            sq.lookup(bad)


def test_ancestors(sq):
    assert sq.ancestors(Code("DCM", "DX")) == [Code("DCM", "DX")]
    for c in CODES:
        a, b = sq.ancestors(c), FAKE.ancestors(c)
        assert a[0] == b[0] == c and set(a) == set(b) and len(a) == len(set(a))
        assert [x.display for x in a] == [x.display for x in b]  # exact BFS order, shortcut rows filtered


def test_ancestors_depth_levels(sq):
    shaft = Code.sct(SHAFT)
    assert Code.sct(ROOT) not in sq.ancestors(shaft)[:5]  # inactive/stated shortcut to root ignored
    assert sq.ancestors(shaft) == FAKE.ancestors(shaft)


def test_finding_groups_and_laterality(sq):
    for c in CODES:
        if c.code in MOCK["sct"]:
            assert sq.finding_groups(c) == FAKE.finding_groups(c)
            assert sq.split_laterality(c) == FAKE.split_laterality(c)
            assert sq.split_laterality(c)[1] == FAKE.split_laterality(c)[1]


def test_procedure_parts(sq):
    for c in LOINCS:
        assert sq.procedure_parts(c) == FAKE.procedure_parts(c)
    with pytest.raises(UnknownCode):
        sq.procedure_parts(Code("LOINC", "0-0"))


def test_find_search(sq):
    for term, root in [("Femur", BODY_STRUCTURE), ("femur", ROOT), ("Fracture", "49755003"), ("Bogus inactive femur term", ROOT),
                       ("Ghost structure", ROOT), ("nothing", ROOT)]:
        assert sq.find(term, root) == FAKE.find(term, root)
    assert sq.find("femur", ROOT)
    for text, root, n in [("shaft of femur", BODY_STRUCTURE, 10), ("femur", BODY_STRUCTURE, 3), ("fracture", "404684003", 10), ("bogus", ROOT, 10)]:
        assert sq.search(text, root, n) == FAKE.search(text, root, n)
    assert [c.code for c in sq.search("shaft of femur", BODY_STRUCTURE)]


def test_import_filters(db):
    con = sqlite3.connect(db[1])
    assert con.execute("SELECT COUNT(*) FROM sct_concept WHERE id = ?", (GHOST,)).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM sct_isa WHERE child = ? AND parent = ?", (SHAFT, ROOT)).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM sct_description WHERE term_lower LIKE 'bogus%'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM loinc_term WHERE loinc = '2345-7'").fetchone()[0] == 0  # non-RAD
    assert con.execute("SELECT COUNT(*) FROM loinc_part WHERE loinc = '2345-7'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM loinc_part_sct WHERE part_number = 'LP15842-5'").fetchone()[0] == 0  # non-SNOMED target
    assert con.execute("SELECT COUNT(*) FROM sct_attr WHERE concept = '71341001' AND type = '272741003'").fetchone()[0] == len(
        [a for a in MOCK["sct"]["71341001"]["attrs"] if a[1] == "272741003"])


def test_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError, match="import-terminology"):
        SqliteTerminology(tmp_path / "nope.db")


def test_resolve_matches_fake(sq):
    inpainters = sup().inpainters
    for req in (CodeRequest(DX, SHAFT_CODE, TRANSVERSE, LEFT, instructions="displaced", seed=1),
                CodeRequest(modality=DX, finding=Code.sct("54441004"), seed=1),
                CodeRequest(procedure=Code("LOINC", "36654-2"), finding=Code.sct("54441004"), seed=1),
                CodeRequest(DX, Code.sct("758654003"), FRACTURE, seed=1),
                CodeRequest(DX, Code.sct("33003004"), FRACTURE, seed=1)):
        a, b = (resolve(req, t, SupportedSet(t, {(DX, SHAFT_CODE)}, set(), inpainters)) for t in (FAKE, sq))
        assert a == b


def test_part_terms():
    from aetiovision.terminology.sqlite import part_terms

    assert next(part_terms("Spine.lumbar")) == "lumbar spine"
    assert {"rib", "hip region", "bone structure", "artery"} <= set(part_terms("Ribs")) | set(part_terms("Hip")) | set(part_terms("Bones")) | set(part_terms("Arteries"))
    t = list(part_terms("Thigh.soft tissue"))
    assert "thigh" in t and "soft tissue" not in t  # coarser parent, never an unrelated generic site
