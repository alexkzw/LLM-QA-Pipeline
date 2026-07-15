#!/usr/bin/env python3
"""Standalone CLI: compare a grounded (RAG) answer against an ungrounded baseline.

One job: for a single question, show the difference between (a) the model
answering with no reference and (b) the grounded + refined RAG pipeline. This
is the qualitative demonstration of why the grounding pipeline matters.

Uses retrieval (top-k chunks), not the full-document path - the full document
path caps out at LLMQA_MAX_REFERENCE_CHARS and can't fit a large PDF like the
OECD reference at all.

Example:
    python scripts/compare_baseline.py --pdf data/oecd_outlook_2026.pdf \
        --question "What is Canada's projected GDP growth rate for 2026?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_qa.config.settings import get_settings
from llm_qa.core.document_loader import load_pdf_text
from llm_qa.core.exceptions import LLMQAError
from llm_qa.core.logging_config import configure_logging, get_logger
from llm_qa.factory import build_rag_pipeline, build_retriever

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Reference PDF to index (skipped if the index already has chunks).",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the index even if it already contains chunks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    try:
        retriever = build_retriever(settings)
        if not retriever.is_indexed or args.force_reindex:
            # Large documents: raise the char ceiling well above the default,
            # since retrieval (not prompt-stuffing) handles the size.
            text = load_pdf_text(args.pdf, max_chars=50_000_000)
            retriever.index_document(text, force=args.force_reindex)

        pipeline = build_rag_pipeline(settings, retriever=retriever)
        baseline = pipeline.answer_without_reference(args.question)
        grounded = pipeline.answer_with_retrieval(args.question)
    except LLMQAError as exc:
        logger.error("Comparison failed: %s", exc)
        return 1

    print("=" * 70)
    print("QUESTION:", args.question)
    print("=" * 70)
    print("\n--- UNGROUNDED BASELINE (model's own knowledge, no document) ---\n")
    print(baseline)
    print("\n--- RAG-GROUNDED + REFINED ANSWER ---\n")
    print(grounded.final_answer)
    print(
        f"\n[grounded={grounded.fully_grounded}, "
        f"iterations={grounded.iterations_used}, "
        f"chunks cited={grounded.retrieved_chunk_ids}]"
    )
    pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
