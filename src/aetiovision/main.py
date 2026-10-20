"""Env vars -> adapters -> Pipeline (D4). The only module importing concrete adapters, lazily."""

from __future__ import annotations

import os
from pathlib import Path

from aetiovision.pipeline.pipeline import Pipeline
from aetiovision.pipeline.ports import AnatomyMasker, CodeExtractor, Inpainter, Terminology

env = os.environ


def data_dir() -> Path:
    return Path(env.get("AETIO_DATA", "data"))


def build() -> Pipeline:
    from aetiovision.catalog.sqlite import SqliteCatalog
    from aetiovision.masking.predefined import PredefinedMasker
    from aetiovision.terminology.sqlite import SqliteTerminology

    data = data_dir()
    terminology = SqliteTerminology(data / "terminology.db")
    return Pipeline(
        terminology=terminology,
        extractor=make_extractor(env.get("AETIO_EXTRACTOR", "rules"), terminology),
        catalog=SqliteCatalog(data / "catalog.db", data),
        masker=PredefinedMasker(fallback=make_mask_fallback(env.get("AETIO_MASK_FALLBACK", "grounded"))),
        inpainters=[make_inpainter(n.strip()) for n in env.get("AETIO_INPAINTERS", "procedural").split(",")],
    )


def make_extractor(name: str, terminology: Terminology) -> CodeExtractor:
    match name:
        case "rules":
            from aetiovision.text_extraction.rules import RulesExtractor

            return RulesExtractor(terminology)
        case "llm":
            from aetiovision.text_extraction.llm import DEFAULT_MODEL, LlmExtractor

            return LlmExtractor(
                terminology,
                provider=env.get("EXTRACTION_PROVIDER", "ollama"),
                model=env.get("EXTRACTION_MODEL", DEFAULT_MODEL),
                base_url=env.get("EXTRACTION_URL"),
            )
    raise SystemExit(f"AETIO_EXTRACTOR: unknown extractor {name!r} (rules | llm)")


def make_mask_fallback(name: str) -> AnatomyMasker | None:
    match name:
        case "none":
            return None
        case "grounded":  # Qwen3-VL via Ollama finds the box, MedSAM worker segments it
            from aetiovision.masking.grounded import GroundedMasker

            return GroundedMasker(
                vlm_model=env.get("DETECTION_MODEL", "qwen3-vl:8b-instruct"),
                ollama_url=env.get("DETECTION_URL", "http://localhost:11434"),
                medsam_url=env.get("MASKING_URL", "http://127.0.0.1:8401"),
            )
    raise SystemExit(f"AETIO_MASK_FALLBACK: unknown masker {name!r} (none | grounded)")


def make_inpainter(name: str) -> Inpainter:
    match name:
        case "procedural":
            from aetiovision.inpainting.procedural import ProceduralInpainter

            return ProceduralInpainter()
        case "diffusion":  # torch lives in the worker process only (D29 revised)
            from aetiovision.inpainting.diffusion import DiffusionInpainter

            return DiffusionInpainter(url=env.get("INPAINTING_URL", "http://127.0.0.1:8402"))
    raise SystemExit(f"AETIO_INPAINTERS: unknown inpainter {name!r} (procedural | diffusion)")


def run_worker(name: str) -> None:
    match name:
        case "medsam":
            from aetiovision.masking.medsam_server import main as serve
        case "diffusion":
            from aetiovision.inpainting.diffusion_server import main as serve
        case _:
            raise SystemExit(f"unknown worker {name!r} (medsam | diffusion)")
    serve()


def import_terminology(snomed: Path, loinc: Path | None) -> None:
    from aetiovision.terminology.importer import import_terminology as run

    run(snomed, loinc, data_dir() / "terminology.db")


def catalog():
    from aetiovision.catalog.sqlite import SqliteCatalog

    return SqliteCatalog(data_dir() / "catalog.db", data_dir())


def main(argv: list[str] | None = None) -> None:
    import sys

    from aetiovision.api.cli import run

    run(sys.modules[__name__], argv)


if __name__ == "__main__":
    main()
