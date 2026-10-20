# AetioVision

Synthetic anomalies on **real healthy** medical images. A request is a code triple: DICOM modality,
SNOMED CT body site and SNOMED CT morphology. A SNOMED finding or a LOINC radiology code also works, and
free text works through step 1. The pipeline picks a healthy template, masks the anatomy, and inpaints
only that region. Exposed as an MCP server (stdio) and a CLI. Research use only.

Design: [`docs/design.html`](docs/design.html). Old text-to-image baselines: [`baselines/`](baselines/README.md).

```
text ─(1 extract)─▶ codes ─(resolve)─▶ triple ─(2 template)─▶ healthy image ─(3 mask)─▶ anatomy mask ─(4 inpaint)─▶ image
```

| Step | Default adapter | Model | Runs in | License |
|---|---|---|---|---|
| 1 text → codes | `rules` (offline) or `llm` | `qwen3:14b` | Ollama | Apache-2.0 |
| resolve | SNOMED CT + LOINC in `data/terminology.db` | – | in-process (SQLite) | SNOMED affiliate / LOINC license |
| 2 template | `data/catalog.db` (FracAtlas hip + hand, normal) | – | in-process (SQLite) | CC BY 4.0 |
| 3 mask | predefined PNG, else **grounded**: VLM box → MedSAM mask | `qwen3-vl:8b-instruct` + `wanglab/medsam-vit-base` | Ollama + worker process | Apache-2.0 / Apache-2.0 |
| 4 inpaint | `procedural` (no ML); `diffusion` optional | SD 1.5 inpainting (or MediSyn UNet) | worker process | CreativeML OpenRAIL-M (MediSyn: CC BY-NC-ND) |

All models are open-weight and run locally. Torch models never load in the MCP/CLI process. They run in
worker processes (`aetio worker medsam|diffusion`) that start automatically on first use and then keep
running. Stop them with `pkill -f "aetiovision.main worker"`.

## Quickstart (macOS, Apple Silicon)

```bash
uv sync --all-extras                          # core + llm + models (torch) extras
ollama pull qwen3-vl:8b-instruct              # masking: anatomy grounding
ollama pull qwen3:14b                         # only for AETIO_EXTRACTOR=llm

# terminology: mock DB from a real SNOMED/LOINC excerpt until you have the release files
uv run python tests/fetch_mock_terminology.py     # excerpt from tx.fhir.org, not in git (license)
uv run python tests/mock_release.py data/releases/mock
uv run aetio import-terminology --snomed data/releases/mock/snomed --loinc data/releases/mock/loinc

# templates: FracAtlas (CC BY 4.0), 20 normal hip + 20 normal hand radiographs
mkdir -p data/raw && curl -L -o data/raw/FracAtlas.zip https://ndownloader.figshare.com/files/65518038 && unzip -q data/raw/FracAtlas.zip -d data/raw
uv run python scripts/ingest_fracatlas.py

uv run aetio supported
uv run aetio codes --finding SCT:54441004 --laterality SCT:24028007 --morphology SCT:70682000 --instructions displaced
uv run aetio text "AP x-ray of the left femoral shaft with a displaced transverse fracture"
AETIO_EXTRACTOR=llm uv run aetio text "Röntgen rechte Hand, Fraktur des fünften Mittelhandknochens"
uv run aetio resolve --procedure LOINC:36654-2 --finding SCT:54441004     # dry run, shows sources + assumptions
```

Results go to `out/`: `<stamp>_<template>_<seed>.png`, plus `_mask.png`, `_template.png` and a `.json` with
the resolved codes, sources, assumptions, template license and the model used. The first run downloads MedSAM
(~375 MB). With `AETIO_INPAINTERS=diffusion` it also downloads SD 1.5 inpainting (~4 GB). Typical times on an
M3 Pro: about 15–30 s with a warm worker, 1–4 min cold.

### MCP (Claude Desktop / Claude Code)

```json
{
  "mcpServers": {
    "aetiovision": {
      "command": "uv",
      "args": ["run", "--directory", "/ABS/PATH/AetioVision", "aetio", "mcp"],
      "env": { "AETIO_EXTRACTOR": "llm", "AETIO_OUT": "/ABS/PATH/AetioVision/out" }
    }
  }
}
```

Claude Code: `claude mcp add aetiovision -- uv run --directory /ABS/PATH/AetioVision aetio mcp`.
Tools: `generate_from_codes`, `generate_from_text`, `resolve_codes`, `list_supported`. Each tool returns a PNG preview,
then JSON with codes and file paths. Errors are returned with `isError` and a typed reason, e.g.
`UnsupportedRequest ... candidates: ...`.

### Configuration (env vars)

| Variable | Values | Default |
|---|---|---|
| `AETIO_DATA` | data dir (terminology.db, catalog.db, images/, masks/) | `./data` |
| `AETIO_OUT` | output dir (MCP) | `./out` |
| `AETIO_EXTRACTOR` | `rules` \| `llm` | `rules` |
| `EXTRACTION_PROVIDER` | `ollama` \| `openai_compat` \| `openai` \| `anthropic` | `ollama` |
| `EXTRACTION_MODEL` | e.g. `qwen3:14b`, `qwen3.5:9b` | `qwen3:14b` |
| `EXTRACTION_URL` | vLLM / llama.cpp / LM Studio / remote Ollama | Ollama at localhost |
| `AETIO_MASK_FALLBACK` | `grounded` \| `none` (predefined masks only) | `grounded` |
| `DETECTION_MODEL`, `DETECTION_URL` | detection: VLM that finds the site box (Ollama) | `qwen3-vl:8b-instruct`, `http://localhost:11434` |
| `MASKING_URL`, `MASKING_MODEL` | masking: box -> mask, SAM-family checkpoint | `http://127.0.0.1:8401`, `wanglab/medsam-vit-base` |
| `AETIO_INPAINTERS` | ordered list, first capable wins: `procedural`, `diffusion` | `procedural` |
| `INPAINTING_URL`, `INPAINTING_BASE`, `INPAINTING_UNET` | inpainting: diffusion worker | `:8402`, `stable-diffusion-v1-5/stable-diffusion-inpainting`, empty (MediSyn: base `stable-diffusion-v1-5/stable-diffusion-v1-5`, unet `hiesingerlab/MediSyn`) |

## Terminology: getting the real release files

The mock DB holds 244 real SNOMED CT concepts and 4 LOINC terms, pulled from the public FHIR test server
`tx.fhir.org` by `tests/fetch_mock_terminology.py`. The mock LOINC part → SNOMED mapping is hand-written.
Replace the mock with the full releases once you have them:

1. **SNOMED CT (RF2).** With your **UMLS account** you can download the *SNOMED CT International Edition* RF2
   zip from NLM: <https://www.nlm.nih.gov/healthit/snomedct/international.html>, sign in with UTS. You can also
   script it with your UTS API key: `curl -L "https://uts-ws.nlm.nih.gov/download?url=<zip url>&apiKey=$UMLS_API_KEY" -o snomed.zip`.
   Licensing: the UMLS license covers SNOMED CT use in SNOMED member countries under the SNOMED affiliate terms,
   and Germany is a member. The official channel in Germany is still the free BfArM affiliate license via MLDS.
   Check with TUM which one you should report under; this is your decision, not something the code can settle.
2. **LOINC.** Get a free account at loinc.org and download the full release zip. It contains `LoincTable/Loinc.csv`
   and `AccessoryFiles/PartFile/*`. UMLS includes LOINC terms but not the part files the importer needs.
3. Unzip both under `data/releases/` (gitignored) and run:
   ```bash
   uv run aetio import-terminology --snomed data/releases/SnomedCT_InternationalRF2_* --loinc data/releases/Loinc_*
   ```
   The importer streams the Snapshot files into `data/terminology.db` and prints LOINC part → SNOMED coverage.
   Unmapped parts raise `UnknownCode` at runtime.

## Keeping licensed data out of git, but usable locally

The rule is that code lives in git and licensed or large data lives in a data directory that git never sees:

| Path | In git? | Why |
|---|---|---|
| `src/`, `tests/`, `scripts/` | yes | code |
| `tests/mock_terminology.json` | **no** | SNOMED/LOINC excerpt for tests: still licensed content, regenerate it (see Tests) |
| `data/catalog.db` | yes (small) | template metadata, points to images by relative path |
| `data/terminology.db`, `data/releases/` | **no** | SNOMED affiliate license forbids redistribution |
| `data/images/`, `data/masks/`, `data/raw/` | **no** | size; each dataset's own license |
| `out/`, HF model cache (`~/.cache/huggingface`) | **no** | generated / downloaded |

- **Locally:** keep `data/` where it is (gitignored), or put it anywhere and set `AETIO_DATA=/path/to/data`,
  e.g. a TUM drive shared by your group.
- **Shipping with Docker:** the image holds code only (`Dockerfile`, `.dockerignore` excludes `data/`).
  `compose.yaml` bind-mounts your local `./data` into the container. Each user builds their own `terminology.db`
  from their own licensed release. Ollama stays on the host (Metal GPU) and is reached via `host.docker.internal`.
  ```bash
  docker compose run --rm aetio supported
  docker compose run --rm -i aetio mcp                      # MCP over stdio from the container
  docker compose --profile models up -d medsam diffusion    # torch workers in containers (Linux/CUDA hosts;
                                                            # on macOS Docker has no GPU, run `aetio worker ...` on the host
                                                            # and set MASKING_URL=http://host.docker.internal:8401)
  ```
- **Terminology server container?** Possible: SNOMED International's open-source **Snowstorm** (needs
  Elasticsearch, ~4–8 GB RAM), or **HAPI FHIR** for LOINC. You still have to load the release files yourself
  under your license; no public image ships SNOMED content. For this project's needs (ancestors, attributes,
  term lookup) the embedded SQLite file (design D12) does the same job as one file with no server. So
  Snowstorm is only worth it if you later need full ECL queries or a shared server for a team.

## Tests

```bash
uv run python tests/fetch_mock_terminology.py   # once: SNOMED/LOINC excerpt from tx.fhir.org (~25 s, network)
uv run pytest -q                                # 50 tests, no network, no models
AETIO_TEST_LLM=1 uv run pytest -q               # + one local-LLM extraction test (Ollama)
```

`tests/mock_terminology.json` (244 SNOMED CT concepts, 4 LOINC terms) is gitignored: the repo is public, and
redistributing SNOMED content worldwide isn't clearly covered by an affiliate license. The fetch script rebuilds it
byte-identical. Without it, the 5 terminology test files are skipped and pytest says so at the end; the masking and
inpainting tests still run. In CI, run the fetch script before pytest.

Includes resolution rules on the fake terminology, SQLite-vs-fake contract tests on the mock release, catalog,
masking QA, inpainters, and the leakage invariant (pixels outside the soft mask never change).

## Status and known limits

- The procedural fracture is a subtle, jagged radiolucent line. "displaced" shifts one fragment, and comminuted
  draws several lines. It's plausible at a glance, but it is not a learned model.
- The `diffusion` inpainter works mechanically. Neither SD 1.5 inpainting nor MediSyn draws a recognisable
  fracture without fine-tuning (design milestone M3: LoRA on FracAtlas fractures).
- Grounded masks follow the bone outline, but may include adjacent bone (the VLM often boxes the whole femur for
  "shaft") or a little soft tissue. Then a displaced fragment shift can leave a visible cut. Predefined masks
  avoid this.
- Templates have no laterality in FracAtlas metadata, so they match any requested side, and the side is chosen by
  radiological convention (patient right = image left). `mirror_ok` mirroring is not implemented.
- Only DX templates (hip, hand) are ingested. CT/MR codes resolve but report `UnsupportedRequest`.
