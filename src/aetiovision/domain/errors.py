class PipelineError(Exception):
    """Base for every expected failure. MCP maps it to isError, CLI to exit code 2."""


class InvalidCode(PipelineError): ...


class UnknownCode(PipelineError): ...


class CodeConflict(PipelineError): ...


class AmbiguousCode(PipelineError): ...


class UnsupportedRequest(PipelineError):
    def __init__(self, message: str, candidates: list[str] | None = None):
        self.candidates = candidates or []
        if self.candidates:
            message += "; candidates: " + ", ".join(self.candidates)
        super().__init__(message)


class NoTemplate(PipelineError): ...


class MaskingFailed(PipelineError): ...


class InpaintFailed(PipelineError): ...
