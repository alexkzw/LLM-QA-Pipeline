#!/usr/bin/env python3
"""Standalone CLI: answer questions against a reference PDF.

One job: take a PDF and one or more questions, run the grounded + refined
pipeline, and print structured results as JSON.

Examples:
    python scripts/run_qa.py --pdf data/oecd_economic_outlook_2026.pdf --question "What is X?"
    python scripts/run_qa.py --pdf data/oecd_economic_outlook_2026.pdf --questions-file qs.txt
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
from llm_qa.factory import build_pipeline

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", required=True, type=Path, help="Path to reference PDF."
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
        reference = load_pdf_text(args.pdf, settings.max_reference_chars)
        pipeline = build_pipeline(settings)
        questions = _load_questions(args)

        results = [
            dataclasses.asdict(pipeline.answer(reference, q)) for q in questions
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
