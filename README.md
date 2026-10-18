# IBM Power Porting Triage Agent

An agentic AI prototype for technical pre-sales and solution architects to qualify customer application migrations to **IBM Power (`ppc64le`)**.

Submit a workload manifest (Dockerfile, `requirements.txt`, SBOM, free-text stack description, or GitHub/doc URL) and the agent streams live reasoning steps while producing an executive feasibility assessment — complete with a readiness score, Fibonacci-sized effort estimate, transitive dependency audit, SIMD/AVX code analysis, and a downloadable PDF memo.

---

## Features

- **Universal Manifest Ingestion** — parses CycloneDX/SPDX SBOMs, Dockerfiles, Python `requirements.txt`, Node `package.json`, Maven `pom.xml`, and free-form text or URLs into a normalized list of `{name, version, ecosystem}` components.
- **Multi-Source Availability Probing** — live checks against Docker Hub, Quay.io and PyPI (ppc64le / universal wheels), plus local IBM index files (ICR Power catalogue, ppc64le wheel index, build-scripts), a curated knowledge base for native/RPM packages, and optional Gemini research. Maven and NPM packages are assumed platform-agnostic; unresolvable PyPI names fall through to later probes rather than being treated as ready.
- **Transitive Dependency Iceberg** — when a package is unported, recursively audits its build-time requirements (`BuildRequires`) and flags any that are also missing on Power.
- **Architecture Sensitivity Scanning** — detects x86 SIMD/AVX intrinsics (`<immintrin.h>`, `__AVX2__`), inline assembly, and 64 KB memory-page-size constraints; recommends SIMDe, Power VSX intrinsics, or allocator configuration as remediation.
- **Fibonacci Effort Sizing** — all person-day estimates are rounded up to the next Fibonacci ceiling (e.g. 5 → 8, 26 → 34) for honest maximum-effort quoting.
- **Live SSE Streaming** — Server-Sent Events surface real-time agent thoughts (IngestionAgent → AvailabilityProber → BuildScopingAgent → ArchitectureHeuristicAgent → ExecutiveSynthesizer).
- **Gemini LLM Integration** — optionally calls Google Gemini for deeper SIMD analysis and executive sales-brief generation; falls back to a deterministic knowledge base with no API key.
- **Repo Scanner** — shallow-clones a GitHub repository to inspect CI matrices, Dockerfiles, and architecture markers.
- **Executive PDF Export** — one-click download of an IBM-branded feasibility memo.
- **IBM Carbon UI** — React + `@carbon/react` dark-themed frontend with filterable dependency tables, architecture audit modal, and CSV/Markdown export.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Browser  (React + Carbon — port 5173)      │
│                                             │
│  InputSection → AgentLiveFeed → Scorecard  │
│  DependencyTable → CodeAuditModal → Export │
└─────────────────┬───────────────────────────┘
                  │  SSE / REST (proxied by Nginx)
┌─────────────────▼───────────────────────────┐
│  FastAPI Backend  (port 8000)               │
│                                             │
│  main.py                                    │
│  ├── /api/health                            │
│  ├── /api/sample-workloads                  │
│  ├── /api/triage          (sync)            │
│  ├── /api/triage/stream   (SSE)             │
│  └── /api/export/pdf                        │
│                                             │
│  agent.py  ← TriageAgent (orchestrator)     │
│    ├── normalizer.py   (manifest parsing)  │
│    ├── prober.py       (registry checks)    │
│    ├── build_analyzer.py (dep iceberg)      │
│    ├── repo_scanner.py (GitHub clone)       │
│    ├── url_extractor.py (URL crawl)         │
│    ├── gemini_helper.py (LLM calls)         │
│    └── pdf_generator.py (ReportLab PDF)     │
└─────────────────────────────────────────────┘
```

---

## Quick Start (Docker)

> Requires Docker (≥ 20) or Podman with `compose`.

```bash
# 1. Clone the repository
git clone <repo-url>
cd power-triage-agent

# 2. Configure environment (optional — runs in simulation mode without a key)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=<your-key>

# 3. Start the stack
./docker-start.sh
```

| Service | URL |
|---|---|
| Web UI | http://localhost:5173 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health |

To stop: `./docker-stop.sh`

---

## Quick Start (Local Development)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# From the repo root:
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(empty)_ | Google Gemini API key. Without it the agent runs in autonomous simulation mode. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model to use. Alternatives: `gemini-3.5-flash-lite`, `gemini-2.5-flash-lite`. |

Copy `.env.example` to `.env` and fill in values before starting the stack.

---

## API Reference

### `POST /api/triage/stream` — SSE Streaming Triage _(recommended)_

Request body (`TriageRequest`):

| Field | Type | Default | Description |
|---|---|---|---|
| `project_name` | string | `"Customer Migration Triage"` | Display name for the report |
| `target_environment` | enum | `rhel9_ocp` | `rhel9_ocp` · `rhel10_ocp` · `rhel9_baremetal` · `rhel10_baremetal` |
| `deliverable_type` | enum | `container` | `container` · `build` · `build_script` |
| `triage_depth` | enum | `deep` | `express` · `deep` |
| `raw_manifest` | string | — | Raw manifest text or a GitHub/doc URL |
| `manifest_type` | string | `auto` | `auto` · `dockerfile` · `requirements` · `sbom` · `text` · `url` |
| `git_repo_url` | string | — | GitHub URL to shallow-clone for repo scan |
| `gemini_api_key` | string | — | Per-request Gemini key override |

Returns a stream of `data: <json>\n\n` SSE events with `type: "step"` (live agent thoughts) followed by a final `type: "result"` event containing the full `TriageResponse`.

### `POST /api/triage` — Synchronous Triage

Same request body as above; waits for the full result and returns it as JSON.

### `POST /api/export/pdf`

Accepts the `TriageResponse` JSON and returns an IBM-branded PDF.

### `GET /api/sample-workloads`

Returns four curated demo workloads ready to paste into the UI.

---

## Demo Workloads

Four built-in presets exercise different triage scenarios:

| Preset | Stack | Key Scenario |
|---|---|---|
| Enterprise Cloud Microservices | Nginx, Redis, PostgreSQL, Kafka, Node.js, Spring Boot | Baseline multi-arch container readiness |
| AI/ML Data Pipeline | PyTorch, NumPy, Pandas, FastAPI, custom C++ AVX2 kernel | SIMD detection + Power VSX remediation |
| High-Throughput Storage Engine | RocksDB + simdjson + Redis | Dependency iceberg + 64 KB page-size risk |
| Legacy Financial Analytics | Intel MKL, NumPy, SciPy | x86 proprietary blocker detection + OpenBLAS/ESSL alternatives |

---

## Project Structure

```
.
├── backend/
│   ├── main.py            # FastAPI app and route handlers
│   ├── agent.py           # TriageAgent — SSE orchestrator
│   ├── models.py          # Pydantic data models
│   ├── normalizer.py      # Universal manifest parser
│   ├── prober.py          # Multi-source registry prober
│   ├── build_analyzer.py  # Transitive dep + SIMD effort engine
│   ├── repo_scanner.py    # Shallow Git clone + arch marker scan
│   ├── url_extractor.py   # URL crawl + LLM dependency inference
│   ├── gemini_helper.py   # Google GenAI wrapper
│   ├── pdf_generator.py   # ReportLab executive PDF generator
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
├── frontend/
│   ├── src/
│   │   └── components/    # React + Carbon components
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── Dockerfile         # Multi-stage: node build → nginx serve
│   └── nginx.conf         # SSE-aware reverse proxy config
├── docker-compose.yml
├── docker-start.sh        # Start stack (Docker or Podman)
├── docker-stop.sh         # Stop stack
└── .env.example
```

---

## Running Tests

```bash
cd backend
source .venv/bin/activate   # or use the repo-root .venv

# From the repo root:
PYTHONPATH=. pytest backend/tests/ -v
```

Covers health check, sample workloads, manifest normalization, registry probing, SIMD/Fibonacci sizing, deliverable-type matching, repo scanning, PDF export, and security/robustness regressions (git URL injection, CORS, SSE error events, CSV/PDF escaping). Some tests call live Docker Hub / PyPI / GitHub and need network access.

---

## Effort Sizing Methodology

Effort estimates use a **Fibonacci ceiling** model:

> Total Effort = BaseBuildEffort + Σ Effort(UnportedBuildDeps) + ArchComplexity + TestValidation

The raw total is buffered by 1 person-week (+5 PD) and scaled by 1.3× to produce a maximum estimate, then rounded up to the next Fibonacci number (e.g. 26 PD → 34 PD). This gives conservative, defensible ceiling quotes for pre-sales conversations.

SIMD effort is further scaled by two multipliers:

- **Instruction count multiplier** — based on the number of x86 SIMD/AVX instructions detected.
- **Complexity multiplier** — one of `DIRECT` (macro drop-in), `SIMDE_COMPATIBLE` (header mapping), `PARTIAL_REWRITE` (mixed VSX rewrite), or `FULL_REDESIGN` (deep AVX-512).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 (Docker image), FastAPI, Uvicorn, Pydantic v2 |
| LLM | Google Gemini (`google-genai`) |
| PDF | ReportLab |
| HTML scraping | BeautifulSoup4, httpx |
| Frontend | React 18, Vite 8, `@carbon/react` |
| Container | Docker / Podman Compose, Nginx |

---

## Known Limitations

- **Curated knowledge base.** Build dependencies, SIMD counts and effort are only reported in detail for packages in a curated table (currently `rocksdb`, `simdjson`). Any other unported package gets a flat 1 PD baseline marked **Low confidence**, with no invented dependencies or SIMD findings; the report states that its build tree, architecture-sensitive code and tests were not assessed and need a source audit. Gemini can refine SIMD sizing when an API key is set and architecture friction is already flagged.
- **Availability is partly asserted.** Native/RPM availability and "verified" versions come from a static table or Gemini unless a live registry probe matches. The static table is not keyed by target OS.
- **Effort is a ceiling.** The headline number applies a +5 PD buffer, a 1.3× factor and a strictly-greater Fibonacci step, so it is intentionally larger than the sum of itemised effort.
- **Security.** Only `https://` repository URLs are cloned (argv-based, 60 s timeout). CORS is limited to local UI origins (override with `CORS_ORIGINS`). Manifests over 500,000 characters are rejected. User-supplied documentation URLs are still fetched server-side, so deploy behind a network egress policy.
