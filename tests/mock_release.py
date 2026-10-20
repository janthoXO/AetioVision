"""Write the mock SNOMED RF2 + LOINC files (real release layout) from mock_terminology.json.

    uv run python tests/mock_release.py data/releases/mock
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

MOCK = json.loads(Path(__file__).with_name("mock_terminology.json").read_text())
INFERRED, STATED = "900000000000011006", "900000000000010007"
FSN, SYN = "900000000000003001", "900000000000013009"
MODULE, DATE = "900000000000207008", "20250201"
# rows the importer must filter out: inactive concept, inactive and stated shortcut IS-A to the root, inactive synonym
GHOST, ROOT, SHAFT, FEMUR = "999000001", "138875005", "41111004", "71341001"


def _tsv(path: Path, header: str, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(header.replace(" ", "\t") + "\r\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\r\n")


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(rows)


def _snomed(out: Path) -> None:
    sct = MOCK["sct"]
    term_dir = out / "Snapshot" / "Terminology"
    concepts = [[c, DATE, 1, MODULE, "900000000000074008"] for c in sct] + [[GHOST, DATE, 0, MODULE, "900000000000074008"]]
    descs, rels = [], []
    n = iter(range(10**6, 10**8))

    def desc(cid, type_id, term, active=1):
        descs.append([next(n), DATE, active, MODULE, cid, "en", type_id, term, "900000020001"])

    def rel(src, dst, grp, type_id, char=INFERRED, active=1):
        rels.append([next(n), DATE, active, MODULE, src, dst, grp, type_id, char, "900000000000451002"])

    for cid, c in sct.items():
        desc(cid, FSN, c["fsn"])
        for s in c["synonyms"]:
            desc(cid, SYN, s)
        for p in c["parents"]:
            rel(cid, p, 0, "116680003")
        for g, t, v in c["attrs"]:
            rel(cid, v, g, t)
    desc(GHOST, FSN, "Ghost structure (body structure)")
    desc(FEMUR, SYN, "Bogus inactive femur term", active=0)
    rel(SHAFT, ROOT, 0, "116680003", active=0)
    rel(SHAFT, ROOT, 0, "116680003", char=STATED)
    rel(FEMUR, "7771000", 1, "272741003", char=STATED)
    _tsv(term_dir / f"sct2_Concept_Snapshot_INT_{DATE}.txt", "id effectiveTime active moduleId definitionStatusId", concepts)
    _tsv(term_dir / f"sct2_Description_Snapshot-en_INT_{DATE}.txt",
         "id effectiveTime active moduleId conceptId languageCode typeId term caseSignificanceId", descs)
    _tsv(term_dir / f"sct2_Relationship_Snapshot_INT_{DATE}.txt",
         "id effectiveTime active moduleId sourceId destinationId relationshipGroup typeId characteristicTypeId modifierId", rels)


def _loinc(out: Path) -> None:
    loinc = MOCK["loinc"]
    _csv(out / "LoincTable" / "Loinc.csv",
         ["LOINC_NUM", "COMPONENT", "PROPERTY", "TIME_ASPCT", "SYSTEM", "SCALE_TYP", "METHOD_TYP", "CLASS", "STATUS", "LONG_COMMON_NAME"],
         [[k, "", "", "", "", "", "", "RAD", "ACTIVE", v["name"]] for k, v in loinc.items()]
         + [["2345-7", "Glucose", "MCnc", "Pt", "Ser/Plas", "Qn", "", "CHEM", "ACTIVE", "Glucose [Mass/volume] in Serum or Plasma"]])
    names = {num: name for v in loinc.values() for _, num, name in v["parts"]}
    _csv(out / "AccessoryFiles" / "PartFile" / "Part.csv", ["PartNumber", "PartTypeName", "PartName", "PartDisplayName", "Status"],
         [[num, t, name, name, "ACTIVE"] for v in loinc.values() for t, num, name in v["parts"]]
         + [["LP15842-5", "COMPONENT", "Glucose", "Glucose", "ACTIVE"]])
    link = ["LoincNumber", "LongCommonName", "PartNumber", "PartName", "PartCodeSystem", "PartTypeName", "LinkTypeName", "Property"]
    _csv(out / "AccessoryFiles" / "PartFile" / "LoincPartLink_Supplementary.csv", link,
         [[k, v["name"], num, name, "http://loinc.org", t, "Search", ""] for k, v in loinc.items() for t, num, name in v["parts"]])
    _csv(out / "AccessoryFiles" / "PartFile" / "LoincPartLink_Primary.csv", link,
         [["2345-7", "Glucose [Mass/volume] in Serum or Plasma", "LP15842-5", "Glucose", "http://loinc.org", "COMPONENT", "Primary", ""]]
         + [["2345-7", "Glucose [Mass/volume] in Serum or Plasma", "LP7057-5", "Serum/Plasma", "http://loinc.org", "SYSTEM", "Primary", ""]])
    sct = MOCK["sct"]
    mapping = ["PartNumber", "PartName", "PartTypeName", "ExtCodeId", "ExtCodeDisplayName", "ExtCodeSystem", "Equivalence",
               "ContentOrigin", "ExtCodeSystemVersion", "ExtCodeSystemCopyrightNotice"]
    _csv(out / "AccessoryFiles" / "PartFile" / "PartRelatedCodeMapping.csv", mapping,
         [[num, names.get(num, ""), "", c, sct[c]["fsn"], "http://snomed.info/sct", "equivalent", "Regenstrief", "2025-02-01", ""]
          for num, c in MOCK["loinc_part_sct"].items()]
         + [["LP15842-5", "Glucose", "COMPONENT", "14749-6", "Glucose", "http://example.org/other", "equivalent", "", "", ""]])


def write_mock_release(out: Path) -> tuple[Path, Path]:
    """Returns (snomed dir, loinc dir)."""
    _snomed(out / "snomed")
    _loinc(out / "loinc")
    return out / "snomed", out / "loinc"


if __name__ == "__main__":
    print(*write_mock_release(Path(sys.argv[1])))
