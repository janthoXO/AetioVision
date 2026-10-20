"""MCP server, stdio (D16). Tools return a PNG preview + JSON with resolved codes and file paths."""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Callable

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from aetiovision.api import service
from aetiovision.api.dto import DCM, LOINC, SCT, GenerateFromCodes
from aetiovision.domain.errors import PipelineError
from aetiovision.pipeline.pipeline import Pipeline

OUT = Path(os.environ.get("AETIO_OUT", "out")).resolve()


def serve(build: Callable[[], Pipeline]) -> None:
    server = MCPServer(
        "aetiovision",
        instructions="Generate synthetic medical images: a real healthy X-ray / CT / MRI slice with a synthetic "
        "anomaly inpainted at a coded body site. Codes: DICOM modality (DCM:DX), SNOMED CT body site and "
        "morphology (SCT:...), or a SNOMED finding / LOINC radiology procedure. Call list_supported first. Research use only.",
    )
    pipeline = cache(build)  # built on first tool call, errors (e.g. missing terminology.db) become tool errors

    def guard(fn):
        try:
            return fn()
        except (PipelineError, ValidationError, FileNotFoundError, RuntimeError) as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    def answer(preview: bytes | None, data: dict) -> list:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        return [Image(data=preview, format="png"), text] if preview else [text]

    @server.tool(structured_output=False)
    def generate_from_codes(
        modality: DCM | None = None,
        body_site: SCT | None = None,
        morphology: SCT | None = None,
        laterality: SCT | None = None,
        finding: SCT | None = None,
        procedure: LOINC | None = None,
        instructions: str = "",
        seed: int | None = None,
    ) -> list:
        """Generate an image from codes: a triple (modality DCM:DX/CT/MR, SNOMED body_site, SNOMED morphology),
        a SNOMED finding, a LOINC radiology procedure, or a mix. laterality: SCT:7771000 left, SCT:24028007 right,
        SCT:51440002 bilateral. instructions: free detail (e.g. "displaced, transverse")."""
        args = dict(modality=modality, body_site=body_site, morphology=morphology, laterality=laterality,
                    finding=finding, procedure=procedure, instructions=instructions, seed=seed)
        return answer(*guard(lambda: service.generate(pipeline(), GenerateFromCodes(**args).to_domain(), OUT)))

    @server.tool(structured_output=False)
    def generate_from_text(text: str, seed: int | None = None) -> list:
        """Generate an image from a free-text request, e.g. "AP x-ray left femur, displaced mid-shaft fracture".
        Returns the extracted and resolved codes too."""
        return answer(*guard(lambda: service.text(pipeline(), text, seed, OUT)))

    @server.tool(structured_output=False)
    def resolve_codes(
        modality: DCM | None = None,
        body_site: SCT | None = None,
        morphology: SCT | None = None,
        laterality: SCT | None = None,
        finding: SCT | None = None,
        procedure: LOINC | None = None,
        instructions: str = "",
    ) -> list:
        """Dry run: resolve codes to the canonical triple without generating. Shows sources and assumptions."""
        args = dict(modality=modality, body_site=body_site, morphology=morphology, laterality=laterality,
                    finding=finding, procedure=procedure, instructions=instructions)
        return answer(None, guard(lambda: service.resolve(pipeline(), GenerateFromCodes(**args))))

    @server.tool(structured_output=False)
    def list_supported() -> list:
        """Supported (modality, body site, morphology) combinations. "any X within REGION" = any SNOMED
        body site of that kind inside the region; a segmentation model locates it."""
        return answer(None, {"supported": guard(lambda: pipeline().supported.listing())})

    server.run("stdio")
