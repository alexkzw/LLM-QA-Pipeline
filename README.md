# Grounded QA Pipeline

A production-grade question-answering system that answers strictly from a supplied reference document, fact-checks every claim, and iteratively refines its answer until no unsupported claims remain.

The supplied gold-standard reference document is the OECD economic outlook for 2026. 3 reasons why this is used as the reference document:

1. It's a 300 page pdf document dense with specific, checkable facts (eg dense tables, precise figures). - this is what makes retrieval & grounding genuinely difficult.

2. It's outside the model's prior knowledge. OECD economic figures such as GDP projections differ from GDP projections via internet searches/news articles. - this allows us to differentiate that the correct answer came from the reference document rather than from the LLM training's data.

3. It's a project where hallucination have real-world consequences. Providing inaccurate GDP projections result in incorrect interest rates set by financial institutions, leading to financial instability and unfair lending practices. 

This project is built with LangChain + Cloudflare Workers AI (Llama 3.1 8B). Originally prototyped in a notebook, refactored into a tested, containerised, deployable service.

---

## What it does

Large language models hallucinate. This pipeline mitigates that with retrieval plus a three-stage grounding loop:

0. **Retrieval (RAG)** — for large documents, the question is embedded and only the most relevant chunks are retrieved from a vector store, so the model sees the few passages that matter instead of a 300-page document.
1. **Grounded generation** — the model answers *only* from the retrieved context, citing passages by number.
2. **Fact-checking** — a validator pass labels every claim as `SUPPORTED`, `UNSUPPORTED`, or `PARTIALLY SUPPORTED`.
3. **Iterative refinement** — if any claim is unsupported, the answer is rewritten using only the evidence, up to a configurable number of passes. The full refinement history and the retrieved chunk IDs are retained for auditability.

The result is an answer with stronger faithfulness guarantees than a single LLM call, plus a transparent record of which sources it used and how it got there.

Two entry points are available: `answer()` (stuff the whole document — fine for short references, via `scripts/run_qa.py`) and `answer_with_retrieval()` (RAG — required for large documents like the 300-page OECD reference, via `scripts/ask_rag.py`).

## Architecture

```
                    ┌─────────────────┐
   reference PDF ──▶│ document_loader │──▶ validated text
                    └─────────────────┘         │
                                                ▼
                                   ┌────────────────────────┐
                                   │   Retriever (RAG)      │
                                   │  chunk → embed → store │   (index once)         
                                   │  (Chroma vector store) │
                                   └────────────────────────┘
                                                │
   question ──────────────────┐   retrieve top-k chunks
                              ▼                 ▼
                    ┌──────────────────────────────────────┐
                    │              QAPipeline              │
                    │  ┌──────────┐   ┌───────────┐        │
                    │  │ generate │──▶│ validate  │──┐     │
                    │  └──────────┘   └───────────┘  │     │
                    │                  ▲         grounded? │
                    │                  │      no ----|     | 
                    │               refine ◀───┘     |     | 
                    │                                │ yes │ 
                    │                     │──────────┘     │ 
                    │                     ▼                │ 
                    │            structured QAResult       │
                    │   (answer + grounded? + chunk IDs)   │
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
│   ├── llm_provider.py        # Cloudflare Workers AI LLM wrapper (retries, timeouts)
│   ├── exceptions.py          # typed error hierarchy
│   └── logging_config.py      # structured logging (text or JSON)
├── retrieval/                 # the RAG layer
│   ├── chunking.py            # sentence-aware overlapping chunker
│   ├── embeddings.py          # local sentence-transformers embeddings (free)
│   ├── vector_store.py        # ChromaDB persistent vector store
│   └── retriever.py           # index + retrieve + format context (with citations)
├── chains/
│   ├── prompts.py             # all prompt templates (incl. RAG prompt) + few-shot
│   └── pipeline.py            # retrieve → generate → validate → refine loop
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

# Compare grounded vs ungrounded answers
python scripts/compare_baseline.py --pdf path/to/short_reference.pdf --question "..."
```

### 6. Run RAG over a large document (the OECD reference)

For large documents — e.g. the 300-page OECD Economic Outlook shipped in
`data/` — use retrieval instead of stuffing the whole document into the
prompt: index once, then ask questions against the index.

```bash
# Index the document into the vector store (chunk -> embed -> persist). Once only.
python scripts/index_document.py --pdf data/oecd_outlook_2026.pdf

# Ask a question via RAG (indexes automatically first if not already indexed)
python scripts/ask_rag.py --pdf data/oecd_outlook_2026.pdf --question "What is the projected global GDP growth rate for 2026?"

# Batch of questions from a file, written to JSON
python scripts/ask_rag.py --pdf data/oecd_outlook_2026.pdf --questions-file questions.txt --output results.json
```

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

- **Why a custom LLM wrapper?** To centralise retries (exponential backoff via `tenacity`), timeouts, and error translation, so transient provider failures don't crash a request.
- **Why return structured `QAResult` objects instead of printing?** So the pipeline is usable from both a CLI and an API, and so refinement history and retrieved chunk IDs can be evaluated and audited.
- **Why local sentence-transformers embeddings + Chroma?** Both run with no API cost and no separate server, so the whole RAG pipeline is free to operate and easy to deploy. Embeddings are normalised so cosine similarity reduces to a dot product.
- **Why chunk with overlap?** A fact split across a chunk boundary would otherwise be unretrievable; the overlap repeats trailing context so boundary-spanning facts survive.
- **Why both an answerable and an adversarial eval set?** Answerable-only accuracy hides the failure mode that matters most for a faithfulness system — confidently answering questions the document doesn't support. The adversarial set measures abstention directly.
- **Why a fake LLM in tests?** Deterministic, free, fast tests that verify the loop logic without depending on a paid external service.
- **Why the iteration cap?** Refinement can loop indefinitely on genuinely unanswerable questions; the cap bounds cost and latency, and the result flags whether grounding fully succeeded.

## Limitations & next steps

- Retrieval uses single-vector dense similarity; adding a re-ranker (cross-encoder) or hybrid keyword+dense retrieval would improve precision on questions with sparse distinctive terms.
- The validator relies on the model's own judgement; pairing it with the retrieved chunks (rather than the generated context) as the ground truth would tighten the faithfulness check further.
- Refinement is sequential; batching independent questions would improve throughput.
- Could add response caching keyed on (document hash, question) to avoid recomputing identical queries.
- Observability could be extended with request tracing (e.g. LangSmith) and per-iteration token accounting.

## License

MIT