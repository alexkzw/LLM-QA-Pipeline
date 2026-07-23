# Deploying to Fly.io, and the ingestion path

This covers two things together because they're operationally linked: the
RAG index lives on a persistent volume, and getting a document onto that
volume in production is a different problem than running
`scripts/index_document.py` on your laptop (see [`README.md`](README.md)
step 6 for the local flow).

## Why Fly.io

Docker-native (builds the existing `Dockerfile` unmodified), supports a
persistent volume and HTTPS with no extra setup, and is cheap/simple enough
for a single-instance portfolio deployment. If you need multi-region or
already live in AWS, ECS Fargate is the more "enterprise" answer, but it's
also a lot more infrastructure (VPC, ALB, IAM, Secrets Manager) to stand up
just to serve one container.

## One-time setup

```bash
# Install flyctl if you haven't: https://fly.io/docs/flyctl/install/
fly auth login

# Claim the app name (fly.toml already has `app = "llm-qa-pipeline"" -
# fly launch will prompt you to rename it if that's taken globally).
fly launch --no-deploy --copy-config

# Create the volume the RAG index will live on. Same region as fly.toml's
# primary_region - a volume is a physical disk attached to one region.
fly volumes create chroma_data --region iad --size 1

# Secrets - never committed, never go in fly.toml.
fly secrets set \
  LLMQA_CLOUDFLARE_API_KEY=your_key_here \
  LLMQA_CLOUDFLARE_ACCOUNT_ID=your_account_id_here \
  LLMQA_API_KEY=$(openssl rand -hex 32)
```

Save that generated `LLMQA_API_KEY` value somewhere - you'll need to send
it as the `X-API-Key` header on every `/ask` request once deployed (see
`api/main.py`'s `require_api_key`), and `fly secrets` won't show it back to
you later.

## Deploy

```bash
fly deploy
```

`fly.toml`'s `[[http_service.checks]]` hits `/health` before routing
traffic to a new machine, the same check the local Docker `HEALTHCHECK` and
CI's `docker-build` smoke test already use - one health-check definition,
exercised in three different places (local container, CI, production).

At this point the app is live but **the vector store is empty** - nothing
has been indexed onto the fresh volume yet. `/ask` (full-document) works
immediately; `answer_with_retrieval` / RAG does not, until you complete the
ingestion step below.

## Ingestion path: getting a document onto the deployed volume

The OECD PDF isn't in the Docker image (`data/*.pdf` is git-ignored - see
`.dockerignore`/`.gitignore` - it's a large third-party binary) and isn't
in the repo either, per the README's note that OECD's site blocks
automated/bot downloads. So the production volume has to be populated
explicitly. Two ways to do it, in order of preference:

### Option A (recommended): index locally, ship the built index

Build the Chroma index on your machine - same command as local dev - then
copy the resulting index directory onto the volume, rather than re-running
embedding (CPU + memory heavy for a 300-page document) on a lean
`shared-cpu-1x` / 1GB production machine.

```bash
# 1. Index locally (skips automatically if already indexed and unchanged -
#    see index_document.py's fingerprinting).
python scripts/index_document.py --pdf data/oecd_outlook_2026.pdf

# 2. Ship the built .chroma/ directory to the volume. flyctl has no rsync
#    equivalent, so this pipes a tar stream over `fly ssh console` - the
#    standard workaround for copying a local directory onto a Fly volume.
tar czf - -C . .chroma | fly ssh console -C "sh -c 'mkdir -p /data && tar xzf - -C /data'"

# 3. Restart so the running process picks up the newly-populated volume
#    (Retriever reads the index at construction time, in the lifespan
#    startup hook - it won't notice files that appear after it's already up).
fly machine restart $(fly machine list --json | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['id'])")
```

To add or update a document later: re-run step 1 (the fingerprint check
means it's a no-op if nothing changed), then repeat steps 2-3. This is the
same idempotent, safe-to-rerun index script the README already documents
for local use - production ingestion reuses it rather than inventing a
second mechanism.

### Option B: index directly on the machine

Simpler to reason about, heavier on the production machine - only worth it
for documents small enough that embedding them on a 1GB shared-CPU machine
isn't painful.

Note `scripts/index_document.py` itself is **not** in the deployed image -
`.dockerignore` excludes `scripts/` (and `data/*.pdf`) from the build
context on purpose, so the image only ships the installed `llm_qa` package.
Call the same underlying library functions the script wraps instead:

```bash
# Put the PDF directly on the volume.
fly ssh sftp shell
> put data/oecd_outlook_2026.pdf /data/oecd_outlook_2026.pdf
> exit

# Index it in place, via the installed package (LLMQA_VECTOR_STORE_DIR is
# already /data/.chroma - see fly.toml) - this is exactly what
# index_document.py itself calls, just invoked directly since the CLI
# script file isn't shipped in the image.
fly ssh console -C "python -c \"
from llm_qa.config.settings import get_settings
from llm_qa.core.document_loader import load_pdf_text
from llm_qa.factory import build_retriever
settings = get_settings()
text = load_pdf_text('/data/oecd_outlook_2026.pdf', max_chars=50_000_000)
count = build_retriever(settings).index_document(text)
print(f'Indexed {count} chunks')
\""
```

### Documented next step, not yet built: an ingestion endpoint

Both options above require `flyctl` access, which is fine for a
single-operator portfolio project but doesn't scale to "a teammate adds a
document without shell access to prod." The natural next step is an
authenticated `POST /admin/reindex` endpoint (reusing the same `X-API-Key`
auth as `/ask`) that accepts an uploaded PDF and calls
`Retriever.index_document` directly - turning this from an ops runbook into
an API call. Not implemented here because it adds a meaningfully different
surface (file upload handling, larger request bodies, a longer-running
request best suited to the same job-status pattern `/ask/async` already
uses) - worth doing once there's more than one person operating this.

## A real constraint worth stating explicitly

A Fly volume mounts to exactly one machine. That means this deployment
cannot be horizontally scaled (multiple machines behind the load balancer)
without moving the vector store off local disk and onto something
network-accessible - a hosted Chroma server, pgvector, or a managed vector
DB (Pinecone, Weaviate). `min_machines_running = 1` in `fly.toml` is
deliberate: this setup is correct for one machine, not a stepping stone to
several without further work.

## Rollback

```bash
fly releases          # list past releases and their image references
fly deploy --image <previous-image-ref>
```

## Post-deploy smoke test

```bash
curl https://llm-qa-pipeline.fly.dev/health

curl -X POST https://llm-qa-pipeline.fly.dev/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <the LLMQA_API_KEY you generated above>" \
  -d '{"reference": "Global growth is projected to slow from 3.4% in 2025 to 2.8% in 2026.", "question": "What is the projected global growth for 2026?"}'
```
