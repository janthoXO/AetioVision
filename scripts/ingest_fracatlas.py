"""Ingest FracAtlas non-fractured, hardware-free, single-scan frontal images into the template catalog."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aetiovision.catalog.sqlite import SqliteCatalog

SOURCE = "FracAtlas, doi:10.6084/m9.figshare.22363012"
# leg and shoulder skipped: leg images are mostly knee/ankle (no clean SNOMED region), add when needed
REGIONS = {"hip": "SCT:61685007", "hand": "SCT:85562004"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracatlas", type=Path, default=Path("data/raw/FracAtlas"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    with open(a.fracatlas / "dataset.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["fractured"] == "0" and r["hardware"] == "0" and r["mixed"] == "0"
                and r["multiscan"] == "0" and r["frontal"] == "1"]
    cat = SqliteCatalog(a.data / "catalog.db", a.data)
    for col, region in REGIONS.items():
        picks = sorted(r["image_id"] for r in rows if r[col] == "1")[: a.n]
        for name in picks:
            path = a.fracatlas / "images" / "Non_fractured" / name
            cat.add(path, "DCM:DX", region, view="frontal", source=SOURCE, license="CC BY 4.0",
                    id=f"fracatlas-{Path(name).stem}")
        print(f"{col}: {len(picks)} templates -> {region}")


if __name__ == "__main__":
    main()
