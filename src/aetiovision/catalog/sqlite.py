"""TemplateCatalog on stdlib sqlite3. Image and mask paths are stored relative to data_dir."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from aetiovision.domain.codes import Code
from aetiovision.domain.image import MedicalImage
from aetiovision.domain.request import ResolvedRequest
from aetiovision.domain.template import Template
from aetiovision.utils.images import load_gray

MAX_CANDIDATES = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS template (
  id TEXT PRIMARY KEY, image TEXT NOT NULL, modality TEXT NOT NULL,
  region TEXT NOT NULL,
  laterality TEXT, view TEXT, mirror_ok INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL, license TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mask (
  template_id TEXT NOT NULL REFERENCES template(id),
  body_site TEXT NOT NULL,
  path TEXT NOT NULL,
  PRIMARY KEY (template_id, body_site)
);
"""


def _resized(im: Image.Image, max_side: int, resample) -> Image.Image:
    s = max_side / max(im.size)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), resample) if s < 1 else im


def _to_gray8(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("I;16", "I"):  # 16-bit: 0.5/99.5 percentile window
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        return Image.fromarray(np.rint(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), "L")
    return im.convert("L")


class SqliteCatalog:
    def __init__(self, db: Path, data_dir: Path):
        self.data_dir = Path(data_dir)
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.executescript(SCHEMA)

    def coverage(self) -> set[tuple[Code, Code]]:
        rows = self.con.execute("SELECT DISTINCT t.modality, m.body_site FROM mask m JOIN template t ON t.id = m.template_id")
        return {(Code.parse(a), Code.parse(b)) for a, b in rows}

    def regions(self) -> set[tuple[Code, Code]]:
        return {(Code.parse(a), Code.parse(b)) for a, b in self.con.execute("SELECT DISTINCT modality, region FROM template")}

    def candidates(self, req: ResolvedRequest, regions: set[Code]) -> list[Template]:
        lat = req.laterality and str(req.laterality)
        # ponytail: mirror_ok ignored, mirroring a template to the other side is not implemented
        rows = self.con.execute(
            "SELECT id, image, modality, region, laterality, view, source, license FROM template "
            "WHERE modality = ? AND (laterality IS NULL OR laterality = ?) ORDER BY id",
            (str(req.supported.modality), lat),
        ).fetchall()
        site = str(req.supported.body_site)
        has_mask = {r[0] for r in self.con.execute("SELECT template_id FROM mask WHERE body_site = ?", (site,))}
        first = [r for r in rows if r[0] in has_mask]
        rest = [r for r in rows if r[0] not in has_mask and Code.parse(r[3]) in regions]
        rng = random.Random(req.seed)
        rng.shuffle(first)
        rng.shuffle(rest)
        return [self._load(r) for r in (first + rest)[:MAX_CANDIDATES]]

    def _load(self, r: tuple) -> Template:
        id, image, modality, region, lat, view, source, license = r
        masks = {f"SCT:{Code.parse(s).code}": self.data_dir / p
                 for s, p in self.con.execute("SELECT body_site, path FROM mask WHERE template_id = ?", (id,))}
        pixels = load_gray(self.data_dir / image)
        return Template(id, MedicalImage(pixels, {"template": id}), Code.parse(modality), Code.parse(region),
                        lat and Code.parse(lat), view, source, license, masks)

    def add(self, image: Path, modality: str, region: str, *, laterality: str | None = None, view: str | None = None,
            source: str, license: str, masks: dict[str, Path] | None = None, id: str | None = None,
            mirror_ok: bool = False, max_side: int = 1024) -> str:
        mod, reg = Code.parse(modality), Code.parse(region)
        lat = laterality and Code.parse(laterality)
        sites = {Code.parse(s): Path(p) for s, p in (masks or {}).items()}
        for c, system in [(mod, "DCM"), (reg, "SCT"), (lat, "SCT"), *((s, "SCT") for s in sites)]:
            if c and c.system != system:
                raise ValueError(f"{c} must be a {system} code")
        id = id or Path(image).stem
        (self.data_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "masks").mkdir(parents=True, exist_ok=True)
        rel = f"images/{id}.png"
        _resized(_to_gray8(Path(image)), max_side, Image.LANCZOS).save(self.data_dir / rel, format="PNG")
        mask_rows = []
        for site, p in sites.items():
            mrel = f"masks/{id}__{site.code}.png"
            _resized(Image.open(p).convert("L"), max_side, Image.NEAREST).save(self.data_dir / mrel, format="PNG")
            mask_rows.append((id, str(site), mrel))
        with self.con:
            self.con.execute("DELETE FROM mask WHERE template_id = ?", (id,))
            self.con.execute(
                "INSERT OR REPLACE INTO template VALUES (?,?,?,?,?,?,?,?,?)",
                (id, rel, str(mod), str(reg), lat and str(lat), view, int(mirror_ok), source, license),
            )
            self.con.executemany("INSERT INTO mask VALUES (?,?,?)", mask_rows)
        return id
