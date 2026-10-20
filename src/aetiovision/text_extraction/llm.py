"""LLM extractor (D18): Pydantic AI over any provider; the model names terms, Terminology maps them to SNOMED."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from aetiovision.domain.codes import BILATERAL, BODY_STRUCTURE, LEFT, MORPHOLOGY, RIGHT, Code
from aetiovision.domain.errors import PipelineError
from aetiovision.domain.request import CodeRequest
from aetiovision.pipeline.ports import Terminology
from aetiovision.pipeline.supported import SupportedSet
from aetiovision.text_extraction.rules import map_term

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
DEFAULT_MODEL = "qwen3:14b"  # 5/5 on the eval prompts; qwen3.5:9b faster but 3/5
OLLAMA_URL = "http://localhost:11434/v1"
SIDES = {"left": LEFT, "right": RIGHT, "bilateral": BILATERAL}

SYSTEM = """You extract structured imaging-request details from a free-text request for a synthetic medical image.
Fill the fields from the text only; use null when the text does not state a value.
- modality: DX (X-ray, radiograph), CT or MR (MRI).
- body_site: the anatomical structure shown, as specific as stated, as a standard SNOMED CT English term
  (e.g. "shaft of femur", "fifth metacarpal bone", "head"). Translate to English. Never put laterality here.
- morphology: the abnormality, as a standard SNOMED CT morphologic abnormality term in English
  (e.g. "transverse fracture", "fracture", "neoplasm"). Not the anatomy.
- laterality: left, right or bilateral.
- instructions: remaining details the codes do not capture (displacement, size, severity, view), in English; empty if none.

The system can generate:
{listing}"""


class Extraction(BaseModel):
    modality: Literal["DX", "CT", "MR"] | None = None
    body_site: str | None = None
    morphology: str | None = None
    laterality: Literal["left", "right", "bilateral"] | None = None
    instructions: str = ""


class LlmExtractor:
    def __init__(self, terminology: Terminology, provider: str, model: str = DEFAULT_MODEL, base_url: str | None = None):
        self.t, self.provider, self.model, self.base_url = terminology, provider, model, base_url
        self._agent = None

    def _build(self):
        from pydantic_ai import Agent

        settings = None
        match self.provider:
            case "ollama":
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.providers.ollama import OllamaProvider

                model = OpenAIChatModel(self.model, provider=OllamaProvider(base_url=self.base_url or OLLAMA_URL))
                settings = {"thinking": False}  # qwen3 reasoning is slow and not needed here
            case "openai_compat":
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.providers.openai import OpenAIProvider

                if not self.base_url:
                    raise PipelineError("EXTRACTION_URL is required for provider openai_compat")
                model = OpenAIChatModel(self.model, provider=OpenAIProvider(base_url=self.base_url))
            case "openai":
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.providers.openai import OpenAIProvider

                model = OpenAIChatModel(self.model, provider=OpenAIProvider(base_url=self.base_url))
            case "anthropic":
                from pydantic_ai.models.anthropic import AnthropicModel
                from pydantic_ai.providers.anthropic import AnthropicProvider

                model = AnthropicModel(self.model, provider=AnthropicProvider(base_url=self.base_url))
            case _:
                raise PipelineError(f"unknown LLM provider {self.provider!r}, use ollama, openai_compat, openai or anthropic")
        return Agent(model, output_type=Extraction, model_settings=settings, retries=2)

    def extract(self, text: str, supported: SupportedSet) -> CodeRequest:
        try:
            if self._agent is None:
                self._agent = self._build()
            prompt = SYSTEM.format(listing="\n".join(f"- {r}" for r in supported.listing()))
            out = self._agent.run_sync(text, instructions=prompt).output
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"LLM extractor ({self.provider}:{self.model}) failed: {e}") from e
        modality = Code("DCM", out.modality) if out.modality else None
        return CodeRequest(
            modality=modality,
            body_site=self._map(supported, out.body_site, BODY_STRUCTURE, modality),
            morphology=self._map(supported, out.morphology, MORPHOLOGY, modality),
            laterality=out.laterality and Code.sct(SIDES[out.laterality]),
            instructions=out.instructions.strip() or text,
        )

    def _map(self, supported, term, root, modality) -> Code | None:
        words = (term or "").lower().split()
        for i in range(len(words)):  # drop leading words: "displaced transverse fracture" -> "transverse fracture"
            if code := map_term(self.t, supported, " ".join(words[i:]), root, modality):
                return code
        return None
