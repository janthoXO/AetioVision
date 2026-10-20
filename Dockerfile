# Code only. Licensed terminology, templates and masks come in through the bind-mounted /data volume.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ARG EXTRAS="--extra llm"
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/venv PATH=/venv/bin:$PATH
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev $EXTRAS
COPY src ./src
RUN uv sync --frozen --no-dev $EXTRAS
ENV AETIO_DATA=/data AETIO_OUT=/out HF_HOME=/cache/hf
ENTRYPOINT ["aetio"]
CMD ["supported"]
