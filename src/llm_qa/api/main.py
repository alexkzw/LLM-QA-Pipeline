"""FastAPI application exposing the QA pipeline as a web service.

Provides:
  * GET  /health          - liveness/readiness probe (Docker & deploy platforms)
  * POST /ask             - grounded QA over a reference + question (synchronous)
  * POST /ask/async       - same, but returns a job id immediately (see api/jobs.py)
  * GET  /ask/jobs/{id}   - poll a job submitted via /ask/async

The pipeline is built once at startup and reused across requests.
"""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from llm_qa import __version__
from llm_qa.api.jobs import JobStore
from llm_qa.api.schemas import (
    AskJobCreated,
    AskJobStatusResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    TokenUsageResponse,
)
from llm_qa.chains.pipeline import QAResult
from llm_qa.config.settings import get_settings
from llm_qa.core.exceptions import LLMProviderError, LLMQAError
from llm_qa.core.logging_config import configure_logging, get_logger
from llm_qa.factory import build_pipeline

logger = get_logger(__name__)

# Holds the singleton pipeline; populated in the lifespan handler.
_state: dict = {}
_job_store = JobStore()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard for endpoints that must not be callable anonymously.

    Compares with secrets.compare_digest rather than `!=` to avoid leaking
    key contents through a response-time side channel. If LLMQA_API_KEY is
    unset, auth is disabled - fine for local dev, never for a deployed
    instance (see Settings.api_key).
    """
    expected = get_settings().api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def _rate_limit_key(request: Request) -> str:
    """Rate-limit per API key when auth is on, else fall back to source IP.

    Keying on the API key (not just IP) is the point: IP-based limiting
    alone is useless behind a shared NAT/corporate proxy, and doesn't stop
    one *authenticated* caller from burning the whole Cloudflare quota in a
    retry loop, which is exactly the failure mode this exists to catch.
    """
    api_key = request.headers.get("x-api-key")
    return api_key if api_key else get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


def _ask_rate_limit() -> str:
    """Read the configured limit at request time, not import time, so
    LLMQA_RATE_LIMIT_PER_MINUTE takes effect without restarting the
    limiter's decorator registration."""
    return f"{get_settings().rate_limit_per_minute}/minute"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build expensive resources once at startup, dispose at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info("Building QA pipeline at startup...")
    _state["pipeline"] = build_pipeline(settings)
    yield
    _state["pipeline"].close()
    _state.clear()


app = FastAPI(
    title="Grounded QA Pipeline",
    description="Answers questions strictly grounded in a supplied reference, "
    "with fact-checked iterative refinement.",
    version=__version__,
    lifespan=lifespan,
)
app.state.limiter = limiter
# slowapi's handler predates Starlette's generic Request[State] signature -
# runtime-correct, just not typed to match exactly.
app.add_exception_handler(
    RateLimitExceeded, _rate_limit_exceeded_handler  # type: ignore[arg-type]
)
app.add_middleware(SlowAPIMiddleware)


def _to_ask_response(question: str, result: QAResult) -> AskResponse:
    usage = result.token_usage
    logger.info(
        "request complete: grounded=%s iterations=%d tokens=%s "
        "estimated_cost_usd=%s",
        result.fully_grounded,
        result.iterations_used,
        usage.total_tokens,
        result.estimated_cost_usd,
    )
    return AskResponse(
        question=question,
        final_answer=result.final_answer,
        fully_grounded=result.fully_grounded,
        iterations_used=result.iterations_used,
        token_usage=TokenUsageResponse(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        ),
        estimated_cost_usd=result.estimated_cost_usd,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(version=__version__)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(_ask_rate_limit)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Answer a question grounded in the supplied reference text.

    Synchronous - the whole validate/refine loop runs before responding.
    For long references or a low max_refinement_iterations budget this can
    take a while; if you need to avoid a client/gateway timeout, use
    POST /ask/async instead.
    """
    pipeline = _state.get("pipeline")
    if pipeline is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="Service not ready.")

    try:
        result = pipeline.answer(payload.reference, payload.question)
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMQAError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_ask_response(result.question, result)


def _run_job(job_id: str, reference: str, question: str) -> None:
    """Executed in Starlette's background threadpool (see BackgroundTasks
    below) - sync code here is fine and does not block the event loop."""
    pipeline = _state.get("pipeline")
    if pipeline is None:  # pragma: no cover - only if startup failed
        _job_store.mark_error(job_id, "Service not ready.")
        return

    _job_store.mark_running(job_id)
    try:
        result = pipeline.answer(reference, question)
        _job_store.mark_done(job_id, result)
    except LLMQAError as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        _job_store.mark_error(job_id, str(exc))


@app.post(
    "/ask/async",
    response_model=AskJobCreated,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(_ask_rate_limit)
def ask_async(
    payload: AskRequest, request: Request, background_tasks: BackgroundTasks
) -> AskJobCreated:
    """Submit a question for background processing; poll /ask/jobs/{id} for
    the result. Prefer this over POST /ask when the caller (or an
    intermediate load balancer/gateway) enforces a request timeout shorter
    than the refine loop might take."""
    if _state.get("pipeline") is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="Service not ready.")

    job = _job_store.create()
    background_tasks.add_task(_run_job, job.id, payload.reference, payload.question)
    return AskJobCreated(job_id=job.id, status=job.status.value)


@app.get(
    "/ask/jobs/{job_id}",
    response_model=AskJobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def get_job(job_id: str) -> AskJobStatusResponse:
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    response_result = (
        _to_ask_response(job.result.question, job.result)
        if job.result is not None
        else None
    )
    return AskJobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        result=response_result,
        error=job.error,
    )
