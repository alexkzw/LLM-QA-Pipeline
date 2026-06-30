"""API request and response schemas.

Explicit, validated contracts for the HTTP layer. Pydantic enforces types and
generates OpenAPI docs automatically.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    reference: str = Field(
        ...,
        min_length=1,
        description="The reference text the answer must be grounded in.",
    )
    question: str = Field(..., min_length=1, description="The question to answer.")


class RefinementStepResponse(BaseModel):
    iteration: int
    is_grounded: bool


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    question: str
    final_answer: str
    fully_grounded: bool
    iterations_used: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
