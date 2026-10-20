"""Build terminology.db (SQLite) from SNOMED CT RF2 and LOINC release directories (design section 3)."""

from __future__ import annotations

import csv
import os
import re
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

from aetiovision.domain.codes import ASSOCIATED_MORPHOLOGY, FINDING_SITE, IS_A, LATERALITY, PART_OF
from aetiovision.domain.errors import UnknownCode

INFERRED = "900000000000011006"
FSN = "900000000000003001"
SCT_URI = "http://snomed.info/sct"
ATTRS = {FINDING_SITE, ASSOCIATED_MORPHOLOGY, LATERALITY, PART_OF}
PART_TYPES = ("Rad.Modality.", "Rad.Anatomic Location.")
SCHEMA = """
CREATE TABLE sct_concept(id TEXT PRIMARY KEY, fsn TEXT, semantic_tag TEXT);
CREATE TABLE sct_description(concept TEXT, term TEXT, term_lower TEXT);
CREATE TABLE sct_isa(child TEXT, parent TEXT);
CREATE TABLE sct_attr(concept TEXT, grp INTEGER, type TEXT, value TEXT);
CREATE TABLE loinc_term(loinc TEXT PRIMARY KEY, long_common_name TEXT);
CREATE TABLE loinc_part(loinc TEXT, part_type TEXT, part_number TEXT, part_name TEXT);
CREATE TABLE loinc_part_sct(part_number TEXT PRIMARY KEY, sct TEXT);
"""
INDEXES = """
CREATE INDEX sct_description_term ON sct_description(term_lower);
CREATE INDEX sct_description_concept ON sct_description(concept);
CREATE INDEX sct_isa_child ON sct_isa(child);
CREATE INDEX sct_isa_parent ON sct_isa(parent);
CREATE INDEX sct_attr_concept ON sct_attr(concept);
CREATE INDEX loinc_part_loinc ON loinc_part(loinc);
"""


def _find(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def _one(root: Path, pattern: str) -> Path:
    found = _find(root, pattern)
    if not found:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return found[0]


def _rf2(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def _csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def _import_snomed(db: sqlite3.Connection, snomed: Path) -> None:
    active = {r["id"] for r in _rf2(_one(snomed, "sct2_Concept_Snapshot*")) if r["active"] == "1"}
    fsns: dict[str, str] = {}

    def descriptions() -> Iterator[tuple[str, str, str]]:
        for r in _rf2(_one(snomed, "sct2_Description_Snapshot-en*")):
            if r["active"] != "1" or r["conceptId"] not in active:
                continue
            term = r["term"]
            if r["typeId"] == FSN:
                fsns[r["conceptId"]] = term
                term = re.sub(r" \([^)]*\)$", "", term)
            yield r["conceptId"], term, term.lower()

    db.executemany("INSERT INTO sct_description VALUES (?,?,?)", descriptions())

    def concepts() -> Iterator[tuple[str, str, str]]:
        for cid in active:
            fsn = fsns.get(cid, "")
            m = re.search(r" \(([^)]*)\)$", fsn)
            yield cid, fsn, m.group(1) if m else ""

    db.executemany("INSERT INTO sct_concept VALUES (?,?,?)", concepts())

    def rels() -> Iterator[dict[str, str]]:
        for r in _rf2(_one(snomed, "sct2_Relationship_Snapshot*")):
            if r["active"] == "1" and r["characteristicTypeId"] == INFERRED:
                yield r

    db.executemany("INSERT INTO sct_isa VALUES (?,?)",
                   ((r["sourceId"], r["destinationId"]) for r in rels() if r["typeId"] == IS_A))
    db.executemany("INSERT INTO sct_attr VALUES (?,?,?,?)",
                   ((r["sourceId"], int(r["relationshipGroup"]), r["typeId"], r["destinationId"])
                    for r in rels() if r["typeId"] in ATTRS))


def _import_loinc(db: sqlite3.Connection, loinc: Path) -> None:
    terms = {r["LOINC_NUM"]: r["LONG_COMMON_NAME"] for r in _csv(_one(loinc, "Loinc.csv")) if r["CLASS"].startswith("RAD")}
    db.executemany("INSERT INTO loinc_term VALUES (?,?)", terms.items())
    for path in _find(loinc, "LoincPartLink_*.csv"):
        db.executemany("INSERT INTO loinc_part VALUES (?,?,?,?)",
                       ((r["LoincNumber"], r["PartTypeName"], r["PartNumber"], r["PartName"]) for r in _csv(path)
                        if r["LoincNumber"] in terms and r["PartTypeName"].startswith(PART_TYPES)))
    db.executemany("INSERT OR IGNORE INTO loinc_part_sct VALUES (?,?)",
                   ((r["PartNumber"], r["ExtCodeId"]) for r in _csv(_one(loinc, "PartRelatedCodeMapping.csv"))
                    if r["ExtCodeSystem"] == SCT_URI))


def _report(db: Path) -> None:
    from aetiovision.terminology.sqlite import SqliteTerminology

    t = SqliteTerminology(db)
    for tbl in ("sct_concept", "sct_description", "sct_isa", "sct_attr", "loinc_term", "loinc_part", "loinc_part_sct"):
        print(f"{tbl}: {t._q(f'SELECT COUNT(*) FROM {tbl}')[0][0]}")
    for kind in ("Region Imaged", "Imaging Focus", "Laterality"):
        part_type = "Rad.Anatomic Location." + kind
        parts = t._q("SELECT DISTINCT part_number, part_name FROM loinc_part WHERE part_type = ?", part_type)
        unmapped = []
        for num, name in parts:
            try:
                t.part_sct(part_type, num, name)
            except UnknownCode:
                unmapped.append(f"{num} {name}")
        mapped = len(parts) - len(unmapped)
        pct = f"{100 * mapped / len(parts):.0f}%" if parts else "n/a"
        print(f"{kind} parts with SNOMED mapping: {mapped}/{len(parts)} ({pct})")
        if unmapped:
            print(f"  unmapped: {', '.join(unmapped[:20])}")
    sys.stdout.flush()


def import_terminology(snomed: Path, loinc: Path | None, db: Path) -> None:
    csv.field_size_limit(sys.maxsize)
    tmp = db.with_suffix(".tmp")
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + SCHEMA)
        with con:  # one transaction
            _import_snomed(con, snomed)
            if loinc is not None:
                _import_loinc(con, loinc)
        con.executescript(INDEXES)
    finally:
        con.close()
    os.replace(tmp, db)
    _report(db)
