#!/usr/bin/env python3
"""Standalone CLI: answer questions via RAG (retrieval-augmented generation).

One job: index a PDF into the vector store if it isn't already indexed, then
answer one or more questions using only the chunks retrieved for each one -
the scalable path for documents too large to stuff into a single prompt.

Examples:
    python scripts/ask_rag.py --pdf data/oecd_outlook_2026.pdf --question "What is X?"
    python scripts/ask_rag.py --pdf data/oecd_outlook_2026.pdf --questions-file qs.txt
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="A single question to answer.")
    group.add_argument(
        "--questions-file",
        type=Path,
        help="Path to a text file with one question per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results (defaults to stdout).",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the index even if it already contains chunks.",
    )
    return parser.parse_args(argv)


def _load_questions(args: argparse.Namespace) -> list[str]:
    if args.question:
        return [args.question]
    lines = args.questions_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


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
        questions = _load_questions(args)

        results = [
            dataclasses.asdict(pipeline.answer_with_retrieval(q)) for q in questions
        ]
    except LLMQAError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    output_json = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        logger.info("Wrote %d result(s) to %s", len(results), args.output)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
