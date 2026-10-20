from __future__ import annotations

from dataclasses import dataclass, field

from aetiovision.domain.codes import Code, Triple


@dataclass(frozen=True)
class CodeRequest:
    """What the client sent, unresolved."""

    modality: Code | None = None
    body_site: Code | None = None
    morphology: Code | None = None
    laterality: Code | None = None
    finding: Code | None = None
    procedure: Code | None = None
    instructions: str = ""
    seed: int | None = None


@dataclass(frozen=True)
class ResolvedRequest:
    requested: Triple  # most specific codes from input, for prompt
    supported: Triple  # nearest supported ancestors, for catalog / mask / inpainter
    laterality: Code | None
    instructions: str
    seed: int
    sources: dict[str, str] = field(default_factory=dict)  # slot -> origin
    assumptions: list[str] = field(default_factory=list)  # slots filled by inference (D19)
