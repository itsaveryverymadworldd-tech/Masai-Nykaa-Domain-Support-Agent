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

## Part 1 — chunking, retrieval and the calibrated threshold

### Task 3 — two chunking strategies, two collections

| | Strategy A: fixed-size + overlap | Strategy B: sentence-based |
| --- | --- | --- |
| Rule | 90-word window, 25-word overlap | whole sentences, 2 per chunk |
| Chunks | 24 | 30 |
| Mean chunk length | 65.7 words | 42.5 words |
| ChromaDB collection | `kb_fixed_overlap` | `kb_sentence` |

Both are embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
normalised) and upserted into **separate** collections with cosine space, so
`distance = 1 - similarity` and the reported similarity is exact.

### Task 4 — empirically calibrated "I don't know" threshold

**No preset is used.** Top-1 cosine similarity was measured for 6 in-scope
queries and 4 deliberately out-of-scope queries, on both collections:

| Collection | in-scope top-1 range | out-of-scope top-1 max | gap | threshold |
| --- | --- | --- | --- | --- |
| A `kb_fixed_overlap` | 0.4632 – 0.8792 | 0.1074 | +0.3559 | 0.2853 |
| B `kb_sentence` | 0.4868 – 0.8748 | 0.1497 | +0.3371 | 0.3182 |

**Deployed threshold: `0.3182`** — the midpoint of the *wider-margin* collection,
and the conservative (higher) choice of the two, because in customer support a
refusal is cheaper than a confident wrong answer.

This is precisely why the brief forbids presets: a tutorial `0.5` would have
**wrongly refused 2 of the 6 in-scope queries** (the return-window query at
0.4632/0.5211 and the escalation query at 0.4868), while `0.3` would have
answered correctly but with almost no margin. The measured clusters sit nowhere
near the common defaults.

Runtime result (`transcripts/04_demo_grounded.txt`): **6/6 in-scope queries
answered from retrieved context with the gold document inside top-3, and 4/4
out-of-scope queries correctly refused** (highest out-of-scope similarity 0.1497).

### Task 5 — precision/recall, both strategies, same 6 queries

Document-level, chunks mapped back to parents and de-duplicated before scoring:

| Collection | mean precision@3 | mean recall@3 | F1 |
| --- | --- | --- | --- |
| A `kb_fixed_overlap` | 0.4722 | 1.0000 | 0.6415 |
| B `kb_sentence` | **0.5556** | **1.0000** | **0.7143** |

Per-query arithmetic for every query and both collections is in
`transcripts/05_eval_chunking.txt`.

**Recommendation: deploy B, the sentence-based strategy.** It beats fixed-overlap
on both axes (P 0.5556 vs 0.4722, R 1.0000 vs 1.0000). The reason is structural:
fixed windows cut through sentences, so a chunk can carry the tail of one policy
claim and the head of the next, diluting its embedding toward whichever half
dominates. Sentence chunks never split a claim — which matters on a policy KB
where a sentence like *"unless the item arrived damaged"* inverts the sentence
before it — and they also gave the cleaner cluster separation in Task 4,
leaving a wider margin around the refusal threshold. **Collection B
(`kb_sentence`) is the fixed input to Parts 2–4.**

---

## Part 2 — CrewAI crew, memory, structured output, guardrails

### Task 6 — `check_order_status` and the designed escalation score

```
escalation_score = status_urgency * (0.40*delayed + 0.30*recency + 0.30*value)

  delayed    1.0 if delayed_shipment else 0.0
  recency    days_since_created / 30          (normalised recency signal)
  value      min(1.0, order_value_inr / 25000) (saturates at the top price band)
  status_urgency: Placed 1.00 | Shipped 1.00 | Returned 0.85 | Delivered 0.55 | Refunded 0.35
```

A weighted blend of three continuous components — **not a bare boolean OR**: two
delayed orders still score differently depending on how stale and how valuable
they are.

**Threshold: `0.3577`, the 80th percentile of `escalation_score` over the
generated dataset** — not a round number. It selects **10/48 orders (20.8%)**,
whose `days_since_created` has median 13 (IQR 8–22): the stalest ~20% of orders.
Delayed orders average **+0.4188** higher than non-delayed ones (0.5472 vs
0.1284). Transcript: `transcripts/06_escalation_score.txt`.

Unknown record ids return an `{"error": "record_not_found"}` dict rather than
raising, so an agent cannot crash the crew by inventing an id.

### Task 7 — the crew, with both tools demonstrably invoked

| Agent | Tools | Role |
| --- | --- | --- |
| Policy Retrieval Specialist | `rag_lookup` | answers policy questions from the KB |
| Order Lookup Specialist | `check_order_status` | **the only** holder of the lookup tool |
| Response Composer | *(none)* | merges both into one `SupportResponse` |

Run via `crew.kickoff()`, sequential process. Transcript `transcripts/07_crew_demo.txt`:

* policy-only query → `rag_lookup` invoked, `check_order_status` **not** invoked
* order query → `rag_lookup` **and** `check_order_status` invoked
* every response validated against the schema

**Tool dispatch is by declared argument schema, never by tool name.** The
transcript prints `dispatched_by = declared_schema` and the schema's required
keys for each call — so a tool named `rag_lookup` can never be misrouted by a
substring test for `"lookup"`.

### Task 8 — session memory

LangChain's `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`, exactly
as the brief prescribes. Two transcripts in `transcripts/08_memory_demo.txt`:

* **Transcript A** (`session-A`): turn 1 *"How long do I have to return a pair
  of shoes?"*, turn 2 *"And what about Home products?"* — memory resolves the
  elliptical follow-up, and the answer correctly gives Home's **10-day** window.
  4 messages stored.
* **Transcript B** (`session-B-fresh`): the same second question in a **fresh**
  session. State is correctly absent — no prior turn, no Home window.

The `LangChainDeprecationWarning` about LangGraph persistence is observed and
**left unsilenced**, as the brief states it is expected and harmless.

### Task 9 — structured output, validated in code

`SupportResponse` (Pydantic) in `src/nykaa/schemas.py`. Live crew responses are
parsed and validated; malformed payloads are **rejected**, because a schema that
never rejects anything proves nothing. Transcript `transcripts/09_schema_demo.txt`:
**3/3 live responses validated, 5/5 malformed payloads rejected** (empty answer,
missing required field, out-of-range `escalation_score`, out-of-range
`top_similarity`, wrong type for `grounded`).

### Task 10 — guardrails, each demonstrated firing

Transcript `transcripts/10_guardrails_demo.txt`:

| Guardrail | Test case | Result |
| --- | --- | --- |
| PII — phone | `9876543210`, `+91 91234 56789`, `09876543210` | all masked |
| PII — card last-4 | `ending in 4321`, `last 4 digits are 8765`, `XXXX-XXXX-XXXX-1357` | all masked |
| Injection | 5 attack patterns + 1 benign control | 5 rejected, control passed |
| Output groundedness | deliberately injected unsupported claim | caught and refused |

The masker is applied to **both** what the model sees and what is written to
disk, so a fixed-format field never reaches a log in the clear.

**Out of scope, as the brief states:** customer name and delivery address are
free text with no reliable pattern under a keyless MOCK_LLM-only masker. They are
recorded in `OUT_OF_SCOPE_PII` in `guardrails.py` as an explicit non-claim rather
than pretend-covered. Only fabricated examples are used anywhere in this repo.

### crewai 1.x adaptations (differences from the brief's 0.x wording)

Probing the installed version (1.15.36) surfaced three differences that produce
**silent** wrong behaviour rather than errors, so they are documented explicitly:

| Brief says | Reality in crewai 1.15.36 | What this repo does |
| --- | --- | --- |
| `response_format` on the Task | **no such field** — `Task.model_fields` shows `output_pydantic` | uses `output_pydantic` **and** validates in code |
| extend `BaseLLM` | correct, and it is a **Pydantic model** — config goes in model fields, and only `call` is abstract | `MockLLM(BaseLLM)` with Pydantic fields |
| (implied) tool calling is automatic | `check_native_tool_support()` needs a callable `supports_function_calling()` or CrewAI silently falls back to text-ReAct | deliberately returns `False` to exercise ReAct, the path the brief's pitfalls concern |

Both documented pitfalls are handled explicitly in `mock_llm.py`:
the tool result is read **only from assistant-authored messages**
(`_observation_from_own_output`), never by scanning the conversation — where the
system prompt's own `Observation:` template would match on the very first call;
and tool arguments are routed by **declared schema** (`_arguments_from_schema`),
never by tool name.

---

## Part 3 — FastAPI deployment, structured logging, evaluation

### Task 11 — FastAPI endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/ask` | one grounded crew turn; input guardrails run first |
| POST | `/add-document` | append a KB document and re-index **both** collections |
| WS | `/ws/chat` | real-time multi-turn chat with per-session memory |
| GET | `/health` | liveness and effective configuration |

Pydantic request/response models on every HTTP endpoint. The WebSocket handler
catches `WebSocketDisconnect` and returns cleanly, so a client dropping
mid-conversation never affects other clients — demonstrated in
`transcripts/11_12_api_and_logging.txt` by having client 1 close the socket
mid-conversation and then serving client 2 successfully afterwards.

**crewai 1.x gotcha found here:** a synchronous `crew.kickoff()` raises
`RuntimeError: Agent execution was invoked synchronously from within a running
event loop` when called from an async handler. The blocking crew is therefore
dispatched with `starlette.concurrency.run_in_threadpool`.

### Task 12 — structured JSON-Lines logging

One JSON object per line, with `trace_id`, `duration_ms` and status, written to
`data/requests.jsonl`. From the transcript: **9 requests → 9 entries, 9/9 with a
trace id and timing.**

The same `mask_pii` used on the model input is applied before writing, so the
log check reports:

```
whole-file search (phone forms)  : NONE
request_text search (card last4) : NONE
[OK] no raw fixed-format PII value reached disk
```

...while the masked form is present, proving masking ran rather than being
skipped: `My number is [PHONE_REDACTED] and the card [CARD_LAST4_REDACTED] was
charged twice.`

### Task 13 — evaluation: 15 queries × four properties

12 in-scope queries (**one per required KB topic**), 1 order-lookup query, and 2
deliberately out-of-scope queries.

| | Accuracy | Grounding | Completeness | Safety |
| --- | --- | --- | --- | --- |
| **Average over 15** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

Per-query scores: `transcripts/13_eval_scores.json`; full transcript with the
evidence behind every score: `transcripts/13_eval_judge.txt`.

The judge is structured as a real LLM judge — a rubric prompt
(`JUDGE_PROMPT` in `src/nykaa/judge.py`) plus a per-case evidence bundle — but
**under MOCK_LLM it is answered by a deterministic rubric evaluator, not a
language model.** That is stated plainly in the module docstring rather than
dressed up: the scores are a reproducible rubric evaluation. Wiring a real model
in would send that prompt instead.

**The evaluation found a real defect.** The first run scored 0.9167, and the
per-query evidence showed why: for two topics (COD refunds, escalation matrix)
the **top-ranked chunk was from the correct document, but the expected fact sat
in a sibling chunk ranked 4th–5th**. The fix is **document expansion** at
generation time — once the top match identifies a document, that document's
other chunks are pulled into the context too. Accuracy then went 0.8667 → 1.0000
and completeness 0.8000 → 1.0000. Task 5's precision/recall is unaffected,
because it measures retrieval *ranking*, which is unchanged.

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
