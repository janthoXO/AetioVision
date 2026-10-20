"""`aetio` command line. Generation commands share the MCP tools' code path (api.service)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from aetiovision.api import service
from aetiovision.api.dto import GenerateFromCodes
from aetiovision.domain.errors import PipelineError

CODE_FLAGS = ("modality", "body_site", "morphology", "laterality", "finding", "procedure")


def _codes_args(p: argparse.ArgumentParser) -> None:
    for f in CODE_FLAGS:
        p.add_argument("--" + f.replace("_", "-"))
    p.add_argument("--instructions", default="")
    p.add_argument("--seed", type=int)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aetio", description="Synthetic anomalies on real healthy medical images.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("codes", "generate from codes"), ("resolve", "resolve codes only (dry run)")):
        _codes_args(sub.add_parser(name, help=help_))
    t = sub.add_parser("text", help="generate from free text (step 1 extracts codes)")
    t.add_argument("text")
    t.add_argument("--seed", type=int)
    t.add_argument("--dry-run", action="store_true", help="extract and resolve only")
    for p in (sub.choices["codes"], t):
        p.add_argument("--out", type=Path, default=Path("out"))
    sub.add_parser("supported", help="list supported triples")
    sub.add_parser("mcp", help="run the MCP server (stdio)")
    it = sub.add_parser("import-terminology", help="SNOMED CT RF2 + LOINC release -> data/terminology.db")
    it.add_argument("--snomed", type=Path, required=True, help="SNOMED CT RF2 release dir")
    it.add_argument("--loinc", type=Path, help="LOINC release dir")
    cat = sub.add_parser("catalog", help="template catalog admin").add_subparsers(dest="catalog_cmd", required=True)
    add = cat.add_parser("add", help="add a healthy template image")
    add.add_argument("image", type=Path)
    add.add_argument("--modality", required=True, help="DCM:DX")
    add.add_argument("--region", required=True, help="SNOMED body structure the image shows")
    add.add_argument("--laterality")
    add.add_argument("--view")
    add.add_argument("--source", required=True)
    add.add_argument("--license", required=True)
    add.add_argument("--mask", action="append", default=[], metavar="SCT:SITE=PNG", help="predefined mask, repeatable")
    add.add_argument("--id")
    w = sub.add_parser("worker", help="run a model worker process (normally auto-started)")
    w.add_argument("name", choices=["medsam", "diffusion"])
    return ap


def run(main, argv: list[str] | None = None) -> None:
    a = _parser().parse_args(argv)
    try:
        match a.cmd:
            case "mcp":
                from aetiovision.api.mcp import serve

                return serve(main.build)
            case "worker":
                return main.run_worker(a.name)
            case "import-terminology":
                return main.import_terminology(a.snomed, a.loinc)
            case "catalog":
                masks = dict(m.split("=", 1) for m in a.mask)
                tid = main.catalog().add(
                    a.image, a.modality, a.region, laterality=a.laterality, view=a.view, source=a.source,
                    license=a.license, masks={k: Path(v) for k, v in masks.items()}, id=a.id,
                )
                return print(f"added template {tid}")
        pipeline = main.build()
        match a.cmd:
            case "supported":
                out = pipeline.supported.listing()
            case "resolve":
                out = service.resolve(pipeline, GenerateFromCodes(**{f: getattr(a, f) for f in (*CODE_FLAGS, "instructions", "seed")}))
            case "codes":
                req = GenerateFromCodes(**{f: getattr(a, f) for f in (*CODE_FLAGS, "instructions", "seed")})
                out = service.generate(pipeline, req.to_domain(), a.out)[1]
            case "text":
                out = service.text(pipeline, a.text, a.seed, None if a.dry_run else a.out)[1]
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except (PipelineError, ValidationError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
