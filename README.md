# Grounded QA Pipeline

A production-grade question-answering system that answers strictly from a supplied reference document, fact-checks every claim, and iteratively refines its answer until no unsupported claims remain.

The supplied gold-standard reference document is the OECD economic outlook for 2026. 3 reasons why this is used as the reference document:

1. It's a 300 page pdf document dense with specific, checkable facts (eg dense tables, precise figures). - this is what makes retrieval & grounding genuinely difficult.

2. It's outside the model's prior knowledge. OECD economic figures such as GDP projections differ from GDP projections via internet searches/news articles. - this allows us to differentiate that the correct answer came from the reference document rather than from the LLM training's data.

3. It's a project where hallucination have real-world consequences. Providing inaccurate GDP projections result in incorrect interest rates set by financial institutions, leading to financial instability and unfair lending practices. 

This project is built with LangChain + Cloudflare Workers AI. Generation uses Llama 3.1 8B; fact-checking uses a three-model majority vote (Mistral, Qwen, and a larger Llama checkpoint) to avoid a validator sharing the same blind spots as its own generator. Originally prototyped in a notebook, refactored into a tested, containerised, deployable service.

---

## What it does

Large language models hallucinate. This pipeline mitigates that with retrieval plus a three-stage grounding loop:

0. **Retrieval (RAG)** — for large documents, the question is embedded and only the most relevant chunks are retrieved from a vector store, so the model sees the few passages that matter instead of a 300-page document. Few-shot examples shown to the generator are also selected dynamically per question (nearest-neighbour over a small curated example bank), rather than the same fixed set injected every time.
1. **Grounded generation** — the model answers *only* from the retrieved context, citing passages by number.
2. **Fact-checking (ensemble)** — three independent, architecturally different models each label every claim as `SUPPORTED`, `UNSUPPORTED`, or `PARTIALLY SUPPORTED` against the same evidence, in parallel; the claim is only accepted as grounded on majority agreement. A validator built from the same weights as the generator tends to miss exactly the mistakes that model makes - independent models reduce that correlated blind spot.
3. **Iterative refinement** - if the majority finds any claim unsupported, the answer is rewritten using only the evidence and every validator's claim-by-claim breakdown (not just one), up to a configurable number of passes. The full refinement history — including each validator's individual verdict — and the retrieved chunk IDs are retained for auditability.

The result is an answer with stronger faithfulness guarantees than a single LLM call, plus a transparent record of which sources it used, which models agreed or disagreed, and how it got there.

Two entry points are available: `answer()` (stuff the whole document — fine for short references, via `scripts/run_qa.py`) and `answer_with_retrieval()` (RAG — required for large documents like the 300-page OECD reference, via `scripts/ask_rag.py`).

## Architecture

```
                    ┌─────────────────┐
   reference PDF ──▶│ document_loader │──▶ validated text
                    └─────────────────┘         │
                                                ▼
                                   ┌────────────────────────┐
                                   │   Retriever (RAG)      │
                                   │  chunk → embed → store │  (index once,
                                   │  (Chroma vector store) │   fingerprinted)
                                   └────────────────────────┘
                                                │
   question ──────────────────┐   retrieve top-k chunks +
                              ▼    dynamic few-shot examples
                    ┌──────────────────────────────────────┐
                    │              QAPipeline              │
                    │  ┌──────────┐   ┌──────────────────┐ │
                    │  │ generate │──▶│ validate (x3, in │ │
                    │  └──────────┘   │ parallel; Llama/ │ │
                    │       ▲         │ Mistral/Qwen vote│ │
                    │       │         └──────────────────┘ │
                    │       │                │   grounded? │
                    │       └── refine ◀── no ┘        yes │
                    │        (sees all 3 verdicts)      │  │
                    │                                   ▼  │
                    │            structured QAResult       │
                    │  (answer + grounded? + chunk IDs +   │
                    │   per-model validator votes)         │
                    └──────────────────────────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                   CLI (scripts/)       FastAPI service     evaluation harness
                                       (POST /ask,/health)  (gold + adversarial)
```

## Project layout

```
src/llm_qa/
├── config/settings.py        # env-based, validated configuration (pydantic-settings)
├── core/
│   ├── document_loader.py     # PDF reading + validation
│   ├── llm_provider.py        # Cloudflare Workers AI LLM wrapper (targeted retries, split timeouts)
│   ├── exceptions.py          # typed error hierarchy
│   └── logging_config.py      # structured logging (text or JSON)
├── retrieval/                 # the RAG layer
│   ├── chunking.py            # sentence-aware overlapping chunker
│   ├── embeddings.py          # local sentence-transformers embeddings (free)
│   ├── vector_store.py        # ChromaDB persistent vector store
│   └── retriever.py           # index + retrieve + format context; fingerprinted
│                               # staleness detection (content/config/model hash)
├── chains/
│   ├── prompts.py             # all prompt templates (incl. RAG prompt) + few-shot
│   ├── grounding.py            # shared groundedness check (single + ensemble validators)
│   ├── example_bank.py         # curated RAG few-shot bank + per-question selector
│   ├── ensemble_validator.py   # 3-model majority-vote fact-checking
│   └── pipeline.py            # retrieve → generate → validate (ensemble) → refine loop
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # request/response models
└── factory.py                 # composition root (build_pipeline / build_rag_pipeline)

scripts/
├── run_qa.py                  # CLI: answer questions against a PDF (full-document, short refs)
├── compare_baseline.py        # CLI: grounded vs ungrounded comparison
├── index_document.py          # CLI: chunk + embed + index a PDF for RAG
├── ask_rag.py                 # CLI: answer questions via RAG (large documents)
└── run_evaluation.py          # CLI: score the RAG pipeline against the gold set

data/
├── evaluation_set.json        # 22 gold-standard + 5 adversarial questions
└── oecd_outlook_2026.pdf      # reference document (CC BY 4.0)

tests/                         # unit + API + retrieval tests (fake LLM, no network)
Dockerfile                     # multi-stage, non-root, healthcheck
docker-compose.yml             # local one-command run
.github/workflows/ci.yml       # lint + type-check + test + docker build
```

## Quickstart

### 1. Set up an environment

Use an isolated environment so dependencies don't collide with your system Python. 

Using `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Or using conda:
```bash
conda create -n llmqa python=3.11 -y
conda activate llmqa
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and set LLMQA_CLOUDFLARE_API_KEY and LLMQA_CLOUDFLARE_ACCOUNT_ID
```

### 4. Get the reference document

The CLI and evaluation harness expect `data/oecd_outlook_2026.pdf`, which isn't committed to git (see `data/*.pdf` in `.gitignore`) since it's a large, third-party binary.

`scripts/download_data.py` exists to automate this, but OECD's site currently sits behind a Cloudflare bot challenge that blocks non-browser requests, so the automated fetch returns `403 Forbidden`. Until that's resolved, download it manually:

1. Open [the OECD Economic Outlook landing page](https://doi.org/10.1787/2d1956f0-en) in a browser and download the PDF.
2. Save it as `data/oecd_outlook_2026.pdf` in your clone of this repo.

Once downloaded, `python scripts/download_data.py` will detect the existing file and skip re-fetching (or re-run it later if OECD's access restrictions change).

### 5. Run the CLI (full-document — short references only)

`run_qa.py` stuffs the *entire* reference into a single prompt, capped at
`LLMQA_MAX_REFERENCE_CHARS` (default 131,072 characters — about 100 pages).
It's for short references you supply yourself; it is **not** suitable for the
300-page OECD PDF shipped in `data/` (see step 6 for that).

```bash
# Single question (swap in your own short PDF)
python scripts/run_qa.py --pdf path/to/short_reference.pdf --question "What is X?"

# Batch of questions from a file, written to JSON
python scripts/run_qa.py --pdf path/to/short_reference.pdf --questions-file questions.txt --output results.json
```

### 6. Run RAG over a large document (the OECD reference)

For large documents — e.g. the 300-page OECD Economic Outlook shipped in
`data/` — use retrieval instead of stuffing the whole document into the
prompt: index once, then ask questions against the index.

```bash
# Index the document into the vector store (chunk -> embed -> persist).
python scripts/index_document.py --pdf data/oecd_outlook_2026.pdf

# Ask a question via RAG (indexes automatically first if not already indexed)
python scripts/ask_rag.py --pdf data/oecd_outlook_2026.pdf --question "What is the projected global GDP growth rate for 2026?"

# Batch of questions from a file, written to JSON
python scripts/ask_rag.py --pdf data/oecd_outlook_2026.pdf --questions-file questions.txt --output results.json
```

`index_document.py` is safe to run every time, on any PDF - there's no need to
remember whether a given document was already indexed. A fingerprint of the
document's content, chunk settings, and embedding model is persisted alongside
the index (`.chroma/index_fingerprint.json`): the same document with the same
settings is detected and skipped automatically; a *different* document (or
changed chunk settings, or a swapped embedding model) is just as automatically
detected and the index is rebuilt - no `--force` needed unless you want to
force a rebuild regardless of whether anything actually changed.

#### Concrete evidence the pipeline is grounded, not guessing

A single grounded answer, on its own, doesn't prove anything — a reader can't
tell whether the number came from retrieval or from a lucky parametric guess.
`compare_baseline.py` runs the *same* question through the model with no
document (its own knowledge) and through the RAG-grounded pipeline, so you can
see whether they diverge:

```bash
python scripts/compare_baseline.py --pdf data/oecd_outlook_2026.pdf \
  --question "What is the projected global GDP growth rate for 2026?"
```

```
--- UNGROUNDED BASELINE (model's own knowledge, no document) ---
I don't have real-time access to the most current economic data, but I can
give you an idea based on available information up to my cut-off date of
December 2023. As per the World Bank's forecast for 2026 (made in 2023), the
global GDP growth rate is expected to be around 3.4%. Similarly, the IMF's
World Economic Outlook (WEO) for 2026 (published in 2023) forecasts a global
GDP growth rate of 3.3%.

--- RAG-GROUNDED + REFINED ANSWER ---
The reference does not provide enough information to answer this question
with a specific rate. However, it mentions that the global growth is
projected to ease to 2.9% in 2026 from 3.4% in 2025 [2], and global trade
growth is projected to moderate from 5.0% in 2025 to 3.1% in 2026 [4].

[grounded=True, iterations=1, chunks cited=[121, 891, 112, 120, 881]]
```

The baseline isn't hallucinating a random number — it's confidently
reciting a **stale, pre-training forecast** (3.4%/3.3%, sourced to 2023) as if
it answered the question about the actual 2026 OECD outlook. That's a more
dangerous failure mode than an obviously wrong answer: it's plausible enough
to trust. The grounded path cites the real, current figure (2.9%) with a
traceable passage number instead.

A second question makes the contrast sharper - a country-specific figure the
baseline has no way to have memorized or guess correctly:

```bash
python scripts/compare_baseline.py --pdf data/oecd_outlook_2026.pdf \
  --question "What is Canada's projected GDP growth rate for 2026?"
```

```
--- UNGROUNDED BASELINE (model's own knowledge, no document) ---
Unfortunately, I cannot verify the most up to date data, but according to
the International Monetary Fund's (IMF) 2022 projections, Canada's
projected GDP growth rate for 2026 is 1.4%.

--- RAG-GROUNDED + REFINED ANSWER ---
Canada's growth is expected to decline from 1.7% in 2025 to 1.2% in 2026
before rebounding to 1.7% in 2027 as domestic demand recovers [4].

[grounded=True, iterations=1, chunks cited=[112, 529, 530, 113, 932]]
```

Same pattern: the baseline cites a different stale figure (1.4%, from a 2022
IMF projection) with unwarranted confidence, while RAG correctly cites 1.2%
against the actual document. Two anecdotal examples aren't proof by
themselves - the systematic version of this same comparison, across 22
gold-standard and 5 adversarial questions, is exactly what `run_evaluation.py`
measures via `answerable_grounded_rate` and `abstention_accuracy` (step 7).

### 7. Run the evaluation harness

This is the project's headline deliverable: a faithfulness benchmark over a
hand-labelled gold set, plus adversarial questions that test whether the system
**abstains** instead of hallucinating.

```bash
python scripts/run_evaluation.py \
  --eval-set data/evaluation_set.json \
  --output eval_results.json
```

It reports three metrics:
- **answerable_accuracy** — fraction of gold questions answered correctly (key-token recall) without declining,
- **answerable_grounded_rate** — fraction where the validator found no unsupported claims,
- **abstention_accuracy** — fraction of adversarial (unanswerable) questions the system correctly refused.

The adversarial set is the most informative number: a system that scores well on
answerable questions but fabricates answers to the adversarial ones is not
production-safe. Reporting both is what separates a real RAG evaluation from a demo.

### 8. Run the API

```bash
make api          # or: uvicorn llm_qa.api.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive OpenAPI docs, or:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"reference": "Global growth is projected to slow from 3.4% in 2025 to 2.8% in 2026 before recovering to 3.1% in 2027.", "question": "What is the projected global growth for 2026?"}'
```

### 9. Run with Docker

```bash
export LLMQA_CLOUDFLARE_API_KEY=your_key_here
export LLMQA_CLOUDFLARE_ACCOUNT_ID=your_account_id_here
docker compose up --build
```

### 10. Deploy to production

See [`DEPLOY.md`](DEPLOY.md) for a live deployment on Fly.io - the persistent
volume, secrets, and the ingestion path for getting a document onto the
deployed index (not the same problem as indexing locally).

## Development

```bash
make check        # run lint + type-check + tests
make lint         # ruff
make type         # mypy
make test         # pytest with coverage
```

Tests run against a **fake LLM**, so they need no API key and make no network calls — the grounding/refinement loop is verified deterministically by injecting scripted responses.

## Configuration reference

All settings are environment variables prefixed `LLMQA_` (or set in `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLMQA_CLOUDFLARE_API_KEY` | *(required)* | Cloudflare API key |
| `LLMQA_CLOUDFLARE_ACCOUNT_ID` | *(required)* | Cloudflare Account ID |
| `LLMQA_MODEL_NAME` | `@cf/meta/llama-3.1-8b-instruct` | Cloudflare Workers AI model identifier |
| `LLMQA_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLMQA_MAX_TOKENS` | `2000` | Max output tokens |
| `LLMQA_MAX_REFINEMENT_ITERATIONS` | `5` | Max refine passes |
| `LLMQA_MAX_REFERENCE_CHARS` | `131072` | Reference size limit (full-document path) |
| `LLMQA_CHUNK_SIZE` | `1000` | Target characters per retrieval chunk |
| `LLMQA_CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `LLMQA_RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question |
| `LLMQA_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `LLMQA_VECTOR_STORE_DIR` | `.chroma` | Vector index persistence directory |
| `LLMQA_LOG_LEVEL` | `INFO` | Log level |
| `LLMQA_LOG_JSON` | `false` | Emit JSON logs |

## Key engineering decisions

- **Why a custom LLM wrapper?** To centralise retries, timeouts, and error translation, so transient provider failures don't crash a request. Retries are *targeted*, not blanket: a predicate (`_is_retryable` in `llm_provider.py`) retries transient failures only - network errors, 429/5xx - not permanent 4xx client errors (bad request, bad auth), which would otherwise burn through the full retry budget failing identically every time. Connect and read timeouts are split, since a slow handshake and a slow generation are different failure signals.
- **Why return structured `QAResult` objects instead of printing?** So the pipeline is usable from both a CLI and an API, and so refinement history, retrieved chunk IDs, and per-model validator votes can be evaluated and audited.
- **Why local sentence-transformers embeddings + Chroma?** Both run with no API cost and no separate server, so the whole RAG pipeline is free to operate and easy to deploy. Embeddings are normalised so cosine similarity reduces to a dot product.
- **Why chunk with overlap?** A fact split across a chunk boundary would otherwise be unretrievable; the overlap repeats trailing context so boundary-spanning facts survive.
- **Why a fingerprint on the index?** "Already indexed" used to mean only "collection non-empty" - it couldn't tell a stale index (different document, different chunk settings, or a swapped embedding model) from a correct one. A persisted fingerprint of (document hash, chunk size/overlap, embedding model name) lets `index_document` detect a mismatch and rebuild automatically instead of silently serving results from the wrong document.
- **Why ensemble (multi-model) validation instead of one model checking its own output?** A validator built from the same weights as the generator tends to share its blind spots - it's grading its own homework. Three architecturally different Cloudflare-hosted models (Llama, Mistral, Qwen) run in parallel and majority-vote on groundedness; disagreement is preserved (not just the winning verdict) so all three claim-by-claim breakdowns feed back into refinement. Different model *weights* matter here, not different hosting providers - a different provider serving the same open-weight model would add zero independence.
- **Why dynamic, per-question few-shot examples instead of a fixed set?** A single hardcoded example set has to generalise to every question shape. Selecting the most relevant example(s) per question (nearest-neighbour over a small curated bank, reusing the same embedding model as retrieval) targets the specific behaviours that are hardest to teach via instructions alone - multi-passage citation format and the exact decline phrasing for unanswerable questions.
- **Why both an answerable and an adversarial eval set?** Answerable-only accuracy hides the failure mode that matters most for a faithfulness system — confidently answering questions the document doesn't support. The adversarial set measures abstention directly.
- **Why a fake LLM (and fake ensemble validator, fake embedder) in tests?** Deterministic, free, fast tests that verify the loop and voting logic without depending on paid external services.
- **Why the iteration cap?** Refinement can loop indefinitely on genuinely unanswerable questions; the cap bounds cost and latency, and the result flags whether grounding fully succeeded.

## Limitations & next steps

- **Ensemble validation triples per-iteration validator cost and calls.** Each refinement iteration now makes 3 validator calls instead of 1 (plus generation/refinement calls) - a real reliability-vs-cost-vs-latency tradeoff, not a free upgrade. Worth measuring against the eval harness before assuming it's a net win on every question type.
- The three validator models are a hardcoded constant (`DEFAULT_VALIDATOR_MODELS` in `ensemble_validator.py`), not yet an env-configurable `Settings` field.
- Retrieval uses single-vector dense similarity; adding a re-ranker (cross-encoder) or hybrid keyword+dense retrieval would improve precision on questions with sparse distinctive terms.
- Chunking (`chunking.py`) has no fallback for a single "sentence" longer than `chunk_size` - a dense table with no sentence-ending punctuation (the OECD document's tables being the obvious risk) could produce one oversized chunk with no hard-truncation safety net.
- Chunk `char_start`/`char_end` offsets are approximate, not exact - reconstructed with a single-space assumption between sentences, so they drift from true source-document positions over a large document. Harmless today (nothing consumes them yet), but not reliable enough to build a "jump to source passage" feature on top of without fixing first.
- `embeddings.py` and `vector_store.py` don't yet wrap their internal failures in the project's typed `LLMQAError` hierarchy the way `retriever.py` and `llm_provider.py` do - a raw exception from either would bypass every CLI script's `except LLMQAError` handling.
- Refinement is sequential; batching independent questions would improve throughput.
- Could add response caching keyed on (document hash, question) to avoid recomputing identical queries.
- Observability could be extended with request tracing (e.g. LangSmith) and per-iteration token/cost accounting - especially valuable now that validation fans out across 3 models.

## License

MIT