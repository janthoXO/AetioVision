"""Use cases shared by the MCP tools and the CLI: run the pipeline, save files, build the JSON answer."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from aetiovision.api.dto import GenerateFromCodes
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.pipeline import Pipeline
from aetiovision.utils.images import preview_png, resolved_to_dict, save_result


def request_to_dict(req: CodeRequest) -> dict:
    values = {f.name: getattr(req, f.name) for f in fields(req)}
    return {k: v if isinstance(v, str | int) else str(v) for k, v in values.items() if v not in (None, "")}


def resolve(pipeline: Pipeline, args: GenerateFromCodes) -> dict:
    return resolved_to_dict(pipeline.resolve(args.to_domain()))


def generate(pipeline: Pipeline, req: CodeRequest, out: Path) -> tuple[bytes, dict]:
    result = pipeline.from_codes(req)
    files = save_result(result, out)
    answer = resolved_to_dict(result.resolved) | {
        "template": {"id": result.template.id, "source": result.template.source, "license": result.template.license},
        "mask": result.mask.source,
        "inpainter": result.inpainter,
        "files": files,
    }
    return preview_png(result.image.pixels), answer


def text(pipeline: Pipeline, text: str, seed: int | None, out: Path | None) -> tuple[bytes | None, dict]:
    req = pipeline.extract(text, seed)
    extracted = {"extracted": request_to_dict(req)}
    if out is None:
        return None, extracted | {"resolved": resolved_to_dict(pipeline.resolve(req))}
    preview, answer = generate(pipeline, req, out)
    return preview, extracted | answer
