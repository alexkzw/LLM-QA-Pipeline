"""In-memory job store backing POST /ask/async + GET /ask/jobs/{id}.

Why a job-status pattern at all: the refine loop can make several
sequential LLM round-trips (generation, up to N iterations of 3 parallel
validator calls plus a refinement call each) - see QAPipeline._refine_against.
A slow question can run past a typical HTTP client or gateway timeout even
though the server is still working correctly. Submitting a job and polling
for its result sidesteps that: the client controls how long it's willing to
wait, independent of any single request's timeout.

Deliberate limitation: this store is a plain dict in one process. Jobs are
lost on restart and invisible across replicas - fine for a single-instance
deployment (see fly.toml), not fine the moment this scales past one
instance. Swapping in Redis (or a DB table) keyed the same way (job id ->
status/result) would remove that limitation without changing the API layer
above this module at all.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from llm_qa.chains.pipeline import QAResult


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: QAResult | None = None
    error: str | None = None


class JobStore:
    """Thread-safe: jobs are written from the background-task thread pool
    (see api/main.py) and read concurrently from polling requests."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.RUNNING

    def mark_done(self, job_id: str, result: QAResult) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.DONE
                job.result = result

    def mark_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.ERROR
                job.error = error
