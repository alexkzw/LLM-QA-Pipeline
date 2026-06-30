#!/usr/bin/env python3
"""Standalone CLI: compare a grounded answer against an ungrounded baseline.

One job: for a single question, show the difference between (a) the model
answering with no reference and (b) the grounded + refined pipeline. This is
the qualitative demonstration of why the grounding pipeline matters.

Example:
    python scripts/compare_baseline.py --pdf data/OECD_economic_outlook_2026.pdf \
        --question "What is the predicted impact of AI on employment?"
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--question", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    try:
        pipeline = build_pipeline(settings)
        baseline = pipeline.answer_without_reference(args.question)

        reference = load_pdf_text(args.pdf, settings.max_reference_chars)
        grounded = pipeline.answer(reference, args.question)
    except LLMQAError as exc:
        logger.error("Comparison failed: %s", exc)
        return 1

    print("=" * 70)
    print("QUESTION:", args.question)
    print("=" * 70)
    print("\n--- UNGROUNDED BASELINE (no reference) ---\n")
    print(baseline)
    print("\n--- GROUNDED + REFINED ANSWER ---\n")
    print(grounded.final_answer)
    print(
        f"\n[grounded={grounded.fully_grounded}, "
        f"iterations={grounded.iterations_used}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
