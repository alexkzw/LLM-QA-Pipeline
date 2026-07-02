#!/usr/bin/env python3
"""Standalone CLI: evaluate the RAG pipeline against the gold-standard set.

One job: run every question in the evaluation set through the RAG pipeline and
report metrics:
  * Answerable questions  -> did the system stay grounded (no UNSUPPORTED) and
    did it surface the gold facts? (faithfulness + a lightweight recall check)
  * Adversarial questions -> did the system correctly DECLINE instead of
    fabricating an answer? (the abstention / anti-hallucination rate)

This separates "looks like it works" from "provably stays faithful".

Example:
    python scripts/run_evaluation.py \
        --eval-set data/evaluation_set.json \
        --output eval_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from llm_qa.config.settings import get_settings
from llm_qa.core.exceptions import LLMQAError
from llm_qa.core.logging_config import configure_logging, get_logger
from llm_qa.factory import build_rag_pipeline

logger = get_logger(__name__)

# Phrases the model is instructed to use when it cannot answer.
_DECLINE_MARKERS = (
    "does not provide enough information",
    "does not contain",
    "cannot be answered",
    "not provide enough information",
    "no information",
    "does not mention",
    "does not specify",
    "does not discuss",
)


def _looks_like_decline(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _DECLINE_MARKERS)


def _key_tokens(gold: str) -> list[str]:
    """Extract salient tokens (numbers and capitalised terms) from a gold answer.

    Used for a lightweight recall check: did the system's answer contain the
    specific figures / named entities the gold answer hinges on? This is a
    heuristic, not a substitute for human review, but it surfaces obvious misses.
    """
    numbers = re.findall(r"\d+\.?\d*%?", gold)
    proper = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", gold)
    return list({*numbers, *proper})


def _recall_score(answer: str, gold: str) -> float:
    tokens = _key_tokens(gold)
    if not tokens:
        return 1.0
    hits = sum(1 for t in tokens if t.lower() in answer.lower())
    return hits / len(tokens)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Where to write JSON results.")
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.5,
        help="Min key-token recall for an answerable item to count as a hit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    eval_data = json.loads(args.eval_set.read_text(encoding="utf-8"))

    try:
        pipeline = build_rag_pipeline(settings)
    except LLMQAError as exc:
        logger.error("Failed to build pipeline: %s", exc)
        return 1

    answerable_results = []
    for item in eval_data.get("answerable", []):
        result = pipeline.answer_with_retrieval(item["question"])
        recall = _recall_score(result.final_answer, item["gold_answer"])
        declined = _looks_like_decline(result.final_answer)
        answerable_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "system_answer": result.final_answer,
                "fully_grounded": result.fully_grounded,
                "key_token_recall": round(recall, 2),
                "passed": (not declined) and recall >= args.recall_threshold,
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
            }
        )

    adversarial_results = []
    for item in eval_data.get("adversarial", []):
        result = pipeline.answer_with_retrieval(item["question"])
        declined = _looks_like_decline(result.final_answer)
        adversarial_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "system_answer": result.final_answer,
                "declined_correctly": declined,
                "passed": declined,  # correct behaviour is to decline
            }
        )

    # --- Aggregate metrics --------------------------------------------
    n_ans = len(answerable_results)
    n_ans_pass = sum(r["passed"] for r in answerable_results)
    n_grounded = sum(r["fully_grounded"] for r in answerable_results)
    n_adv = len(adversarial_results)
    n_adv_pass = sum(r["passed"] for r in adversarial_results)

    summary = {
        "answerable_total": n_ans,
        "answerable_passed": n_ans_pass,
        "answerable_accuracy": round(n_ans_pass / n_ans, 3) if n_ans else None,
        "answerable_grounded_rate": round(n_grounded / n_ans, 3) if n_ans else None,
        "adversarial_total": n_adv,
        "adversarial_passed": n_adv_pass,
        "abstention_accuracy": round(n_adv_pass / n_adv, 3) if n_adv else None,
    }

    output = {
        "summary": summary,
        "answerable": answerable_results,
        "adversarial": adversarial_results,
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        logger.info("Wrote results to %s", args.output)

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
