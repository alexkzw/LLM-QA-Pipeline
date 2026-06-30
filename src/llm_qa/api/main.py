"""FastAPI application exposing the QA pipeline as a web service.

Provides:
  * GET  /health  - liveness/readiness probe (used by Docker & deploy platforms)
  * POST /ask     - grounded QA over a supplied reference + question

The pipeline is built once at startup and reused across requests.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from llm_qa import __version__
from llm_qa.api.schemas import AskRequest, AskResponse, HealthResponse
from llm_qa.config.settings import get_settings
from llm_qa.core.exceptions import LLMProviderError, LLMQAError
from llm_qa.core.logging_config import configure_logging, get_logger
from llm_qa.factory import build_pipeline

logger = get_logger(__name__)

# Holds the singleton pipeline; populated in the lifespan handler.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build expensive resources once at startup, dispose at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info("Building QA pipeline at startup...")
    _state["pipeline"] = build_pipeline(settings)
    yield
    _state.clear()


app = FastAPI(
    title="Grounded QA Pipeline",
    description="Answers questions strictly grounded in a supplied reference, "
    "with fact-checked iterative refinement.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(version=__version__)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question grounded in the supplied reference text."""
    pipeline = _state.get("pipeline")
    if pipeline is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="Service not ready.")

    try:
        result = pipeline.answer(request.reference, request.question)
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMQAError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        question=result.question,
        final_answer=result.final_answer,
        fully_grounded=result.fully_grounded,
        iterations_used=result.iterations_used,
    )
