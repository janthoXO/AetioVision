"""Regenerate tests/mock_terminology.json: a small, real excerpt of SNOMED CT + LOINC.

Pulled from the public FHIR test server tx.fhir.org (SNOMED CT International, LOINC) for
the seed codes below plus every ancestor (IS-A + Part of) and attribute target. Used by
tests/fakes.py and tests/mock_release.py until the real release files are imported.
LOINC part -> SNOMED mapping is hand-written here (PartRelatedCodeMapping is not on FHIR).

    uv run python tests/fetch_mock_terminology.py
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TX = "https://tx.fhir.org/r4/CodeSystem/$lookup?system={}&code={}&property=*"
SCT, LNC = "http://snomed.info/sct", "http://loinc.org"
ROOT = "138875005"
KEEP_ATTRS = {"363698007", "116676008", "272741003", "123005000"}

SEED_SCT = [
    # body structures
    "41111004", "33003004", "182053004", "758654003", "758655002", "71341001", "29627003",
    "12611008", "62413002", "36455000", "50939004", "61685007", "53120007", "69536005",
    # morphologies
    "72704001", "70682000", "7905002", "13321001", "108369006",
    # findings
    "71620000", "54441004", "1303377000", "12676007", "208393000", "31978002",
    # laterality
    "7771000", "24028007", "51440002",
]
SEED_LOINC = ["24725-4", "36654-2", "36653-4", "26389-7"]
# LOINC radiology part -> SNOMED (mock of PartRelatedCodeMapping.csv), filled after part names are known
PART_SCT_BY_NAME = {
    "Head": "69536005", "Femur": "71341001", "Hand": "85562004", "Left": "7771000",
    "Right": "24028007", "Bilateral": "51440002", "Lower extremity": "61685007", "Upper extremity": "53120007",
}
RAD_PROPS = {
    "rad-modality-modality-type": "Rad.Modality.Modality Type",
    "rad-anatomic-location-region-imaged": "Rad.Anatomic Location.Region Imaged",
    "rad-anatomic-location-imaging-focus": "Rad.Anatomic Location.Imaging Focus",
    "rad-anatomic-location-laterality": "Rad.Anatomic Location.Laterality",
}


def lookup(system: str, code: str) -> dict:
    req = urllib.request.Request(TX.format(system, code), headers={"Accept": "application/fhir+json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def props(d: dict):
    for p in d.get("parameter", []):
        if p["name"] == "property":
            q = {x["name"]: x for x in p["part"]}
            v = q["value"]
            yield q["code"]["valueCode"], next(val for k, val in v.items() if k != "name")
        elif p["name"] == "display":
            yield "display", p["valueString"]
        elif p["name"] == "designation":
            q = {x["name"]: x for x in p["part"]}
            if q.get("status", {}).get("valueCode") != "inactive" and q.get("language", {}).get("valueCode", "en") == "en":
                yield "designation:" + q["use"]["valueCoding"]["code"], q["value"]["valueString"]


def sct_concept(code: str) -> dict:
    c = {"fsn": "", "synonyms": [], "parents": [], "attrs": []}
    for k, v in props(lookup(SCT, code)):
        if k == "designation:900000000000003001":
            c["fsn"] = v
        elif k == "designation:900000000000013009":
            c["synonyms"].append(v)
        elif k == "parent":
            c["parents"].append(v)
        elif k in KEEP_ATTRS:
            c["attrs"].append([1, k, v])  # FHIR lookup flattens role groups: mock puts all in group 1
    return c


def main() -> None:
    sct: dict[str, dict] = {}
    todo = set(SEED_SCT) | {"138875005", "123037004", "49755003", "404684003", "85562004"}
    with ThreadPoolExecutor(8) as pool:
        while todo:
            batch = sorted(todo - sct.keys())
            todo = set()
            for code, c in zip(batch, pool.map(sct_concept, batch)):
                sct[code] = c
                todo |= set(c["parents"]) | {v for _, t, v in c["attrs"]}
            todo -= sct.keys()
            print(f"{len(sct)} concepts")

    loinc, part_names = {}, {}
    for num in SEED_LOINC:
        d = dict(name="", parts=[])
        for k, v in props(lookup(LNC, num)):
            if k == "display":
                d["name"] = v
            elif k in RAD_PROPS:
                if v not in part_names:
                    part_names[v] = next(x for kk, x in props(lookup(LNC, v)) if kk == "display")
                d["parts"].append([RAD_PROPS[k], v, part_names[v]])
        loinc[num] = d
    part_sct = {p: PART_SCT_BY_NAME[n] for p, n in part_names.items() if n in PART_SCT_BY_NAME}
    out = {"_source": "tx.fhir.org SNOMED CT International + LOINC, excerpt for tests", "sct": sct, "loinc": loinc, "loinc_part_sct": part_sct}
    Path(__file__).with_name("mock_terminology.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {len(sct)} SNOMED concepts, {len(loinc)} LOINC terms, parts {part_names}")


if __name__ == "__main__":
    main()
