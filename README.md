# Nykaa Domain Support Agent — Final Capstone

**Track: E-commerce & Retail (Nykaa).** This single public repository contains the
complete capstone: dataset, RAG core, CrewAI agent crew, Autogen review stage,
governance layer, and FastAPI deployment.

> **Everything in this repository runs under `MOCK_LLM` with zero API keys and
> zero outbound network access.** No paid account, credit card, GPU, or API key
> is required at any point. Full details in [LLM configuration](#llm-configuration-mock_llm).

---

## Quick start

```bash
# 1. Install uv (no root required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install every dependency
uv sync

# 3. Verify the stack (imports, CPU-only torch, telemetry disabled)
uv run python scripts/00_verify_env.py

# 4. Regenerate the dataset and print the Task 1 report
uv run python dataset.py
```

`uv.lock` is committed, so `uv sync` reproduces the exact dependency set.

Python is pinned to **3.12** via `.python-version`. Python 3.14 (current Arch
default) is deliberately *not* used: several dependencies in this stack have no
3.14 wheels yet.

---

## Part 1, Task 1 — dataset design choices

`dataset.py` is a seeded, deterministic generator. The grader can reproduce the
dataset exactly by running `uv run python dataset.py`. The choices made, as
required:

| Design choice | Value |
| --- | --- |
| **Seed** | `20261106` |
| **Number of records** | `48` (requirement: ≥ 40) |
| **Category weights** | Apparel `0.28`, Beauty `0.30`, Footwear `0.16`, Home `0.14`, Electronics `0.12` |
| **Status weights** | Delivered `0.44`, Shipped `0.22`, Placed `0.14`, Returned `0.11`, Refunded `0.09` |
| **Amount range (INR)** | `349` – `24,999` overall, drawn **log-uniform** inside a per-category band |
| **`days_since_created`** | uniform integer in `[0, 30]` |
| **Delay model** | per-status probability (`DELAY_PROB_BY_STATUS`), not a flat rate |

**Per-category price bands (INR).** Beauty `349–4,999`, Apparel `499–5,999`,
Footwear `799–6,999`, Home `399–7,999`, Electronics `1,499–24,999`.

**Reasoning for the amount range.** Nykaa's catalogue runs from a ~₹349 lipstick
to a ~₹24,999 laptop, and spend is right-skewed — many low-value orders, few very
expensive ones — so a **log-uniform** draw inside each category's own band models
the real distribution far better than a flat uniform draw over one global range.

**How the structural thresholds are guaranteed without hand-editing records.**

* *Category coverage (≥3 each)* and *status coverage (≥1 each)* are guaranteed
  **by construction**: `_stratified_values` first reserves each value's minimum,
  then distributes the remaining slots by the weights above using the
  largest-remainder method. Coverage is therefore structural, never patched in.
* *The `delayed_shipment` band (10–30%)* is the one genuinely random threshold.
  It is **not** forced by editing records. Each status carries its own plausible
  delay probability, the realised rate is measured, and `find_seed_in_band()` —
  committed in `dataset.py` so the choice is auditable — was used at development
  time to select the seed `20261106`, exactly the "change your seed and
  regenerate" remedy the brief prescribes. `generate_orders()` re-validates the
  band on every call and raises if a later weight edit breaks it.

**Measured output at seed `20261106`** (reproduced by `uv run python dataset.py`):

```
total records             : 48   (required >= 40)
count per category        : Apparel 12 | Beauty 13 | Electronics 7 | Footwear 8 | Home 8   (each >= 3)
count per status          : Placed 7 | Shipped 10 | Delivered 20 | Returned 6 | Refunded 5   (each >= 1)
delayed_shipment=True     : 8/48 = 16.7%   (required 10%-30%)
order_value_inr range     : Rs.376 - Rs.22,660
days_since_created range  : 0 - 29   (required 0-30)
```

---

## LLM configuration (`MOCK_LLM`)

The whole project defaults to **`MOCK_LLM`**: a deterministic, offline
implementation that needs no API key and makes no network call. **This is the
mode every graded transcript in this repository uses.**

A real LLM API may optionally be wired in behind an environment-variable flag,
but every acceptance criterion is satisfied under `MOCK_LLM` alone.

**Telemetry is explicitly disabled**, since both CrewAI and ChromaDB attempt
outbound calls by default and would silently violate the zero-network
requirement. `src/nykaa/config.py` sets these at import time, *before* `crewai`
or `chromadb` is imported anywhere in the process:

```
CREWAI_DISABLE_TELEMETRY = true
OTEL_SDK_DISABLED        = true
ANONYMIZED_TELEMETRY     = False
CHROMA_TELEMETRY_DISABLED= True
HF_HUB_OFFLINE           = 1
TRANSFORMERS_OFFLINE     = 1
```

**Confirmed: `CREWAI_DISABLE_TELEMETRY=true` (and `OTEL_SDK_DISABLED=true`) is
set as an environment variable before the crew runs.**

---

## Repository layout

```
dataset.py              Task 1 — deterministic order dataset (pure stdlib)
kb/                     Task 2 — the ≥12 knowledge-base documents
src/nykaa/              RAG core, crew, review stage, governance, API
scripts/                Runnable demos; each writes a transcript
transcripts/            Captured evidence for every acceptance criterion
data/chroma/            Generated vector store (gitignored, rebuilt by scripts)
```

---

## Environment

Verified by `scripts/00_verify_env.py`:

| Component | Version | Role |
| --- | --- | --- |
| Python | 3.12.16 | pinned via `.python-version` |
| torch | 2.15.0**+cpu** | CPU-only build — see note below |
| sentence-transformers | 6.2.0 | local embeddings, no API key |
| chromadb | 1.5.9 | vector index, two collections |
| crewai | 1.15.36 | Part 2 orchestration |
| langchain-core | 1.7.6 | Part 2 session memory |
| autogen-agentchat | 0.7.5 | Part 4 review stage |
| fastapi / uvicorn | 0.162.0 / 0.54.0 | Part 3 deployment |
| pydantic | 2.12.5 | structured outputs |

**CPU-only PyTorch.** `sentence-transformers` pulls in `torch`, and PyPI's
default Linux wheel drags in ~7 GB of CUDA runtime libraries (`nvidia-cu13-*`)
that this project never uses. `pyproject.toml` therefore declares `torch` as a
direct dependency and maps it to PyTorch's dedicated CPU index via
`[tool.uv.sources]`, which drops the install to **0 CUDA packages** and removes
any GPU requirement. Two gotchas worth recording:

1. `[tool.uv.sources]` only applies to *direct* dependencies — leaving `torch`
   transitive through `sentence-transformers` silently resolves the CUDA build.
2. The mapping only takes effect on a **fresh** resolution. A stale `uv.lock` is
   reused verbatim, so delete it (or run `uv lock -U`) after editing the
   mapping.
