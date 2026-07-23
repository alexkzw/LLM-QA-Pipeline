"""API-layer tests using FastAPI's TestClient and a stubbed pipeline."""
from __future__ import annotations

from types import SimpleNamespace

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


def test_ask_endpoint_rejects_missing_or_wrong_api_key(monkeypatch) -> None:
    api_main._state["pipeline"] = StubPipeline()
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(api_key="secret123", rate_limit_per_minute=100),
    )
    client = TestClient(api_main.app)
    payload = {"reference": "some reference", "question": "what is x?"}

    response = client.post("/ask", json=payload)
    assert response.status_code == 401

    response = client.post("/ask", json=payload, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_ask_endpoint_accepts_correct_api_key(monkeypatch) -> None:
    api_main._state["pipeline"] = StubPipeline()
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(api_key="secret123", rate_limit_per_minute=100),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/ask",
        json={"reference": "some reference", "question": "what is x?"},
        headers={"X-API-Key": "secret123"},
    )
    assert response.status_code == 200
    assert response.json()["final_answer"] == "grounded answer"


def test_ask_async_job_completes_and_is_pollable() -> None:
    api_main._state["pipeline"] = StubPipeline()
    client = TestClient(api_main.app)

    submitted = client.post(
        "/ask/async",
        json={"reference": "some reference", "question": "what is x?"},
    )
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]

    # TestClient runs background tasks to completion within the same call,
    # so the job should already be done by the time we poll - no sleep loop.
    polled = client.get(f"/ask/jobs/{job_id}")
    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "done"
    assert body["result"]["final_answer"] == "grounded answer"
    assert body["error"] is None


def test_ask_jobs_unknown_id_returns_404() -> None:
    api_main._state["pipeline"] = StubPipeline()
    client = TestClient(api_main.app)
    response = client.get("/ask/jobs/does-not-exist")
    assert response.status_code == 404
