"""API-layer tests using FastAPI's TestClient and a stubbed pipeline."""
from __future__ import annotations

from fastapi.testclient import TestClient

from llm_qa.api import main as api_main
from llm_qa.chains.pipeline import QAResult


class StubPipeline:
    def answer(self, reference: str, question: str) -> QAResult:
        return QAResult(
            question=question,
            initial_answer="init",
            final_answer="grounded answer",
            iterations_used=1,
            fully_grounded=True,
        )


def test_health_endpoint() -> None:
    api_main._state["pipeline"] = StubPipeline()
    client = TestClient(api_main.app)
    # Call the route function directly to avoid lifespan network setup.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_returns_grounded_answer() -> None:
    api_main._state["pipeline"] = StubPipeline()
    client = TestClient(api_main.app)
    response = client.post(
        "/ask",
        json={"reference": "some reference", "question": "what is x?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["final_answer"] == "grounded answer"
    assert body["fully_grounded"] is True


def test_ask_endpoint_validates_empty_question() -> None:
    api_main._state["pipeline"] = StubPipeline()
    client = TestClient(api_main.app)
    response = client.post("/ask", json={"reference": "r", "question": ""})
    assert response.status_code == 422  # pydantic validation error
