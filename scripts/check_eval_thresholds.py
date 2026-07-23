#!/usr/bin/env python3
"""Standalone CLI: fail (exit 1) if run_evaluation.py's summary metrics
dropped below configured thresholds.

Separate from run_evaluation.py on purpose: the evaluation run and the
pass/fail judgement on its output are different concerns - run_evaluation.py
can be used interactively (a human reads the numbers) or as a CI gate (this
script). Keeping the threshold check standalone means it can be re-run
against a saved eval_results.json without paying for another live model run.

Example:
    python scripts/check_eval_thresholds.py --results eval_results.json \
        --min-answerable-accuracy 0.7 \
        --min-answerable-grounded-rate 0.8 \
        --min-abstention-accuracy 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--min-answerable-accuracy",
        type=float,
        default=0.7,
        help="Minimum fraction of answerable questions answered correctly.",
    )
    parser.add_argument(
        "--min-answerable-grounded-rate",
        type=float,
        default=0.8,
        help="Minimum fraction of answerable questions accepted as grounded.",
    )
    parser.add_argument(
        "--min-abstention-accuracy",
        type=float,
        default=0.8,
        help="Minimum fraction of adversarial questions correctly declined.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        default=True,
        help="Fail if the run stopped early (e.g. quota exhausted) instead "
        "of covering the full eval set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = json.loads(args.results.read_text(encoding="utf-8"))
    summary = data["summary"]

    numeric_checks = [
        ("answerable_accuracy", args.min_answerable_accuracy),
        ("answerable_grounded_rate", args.min_answerable_grounded_rate),
        ("abstention_accuracy", args.min_abstention_accuracy),
    ]

    print("=== EVAL THRESHOLD CHECK ===")
    all_passed = True

    if args.require_complete:
        complete = summary.get("complete") is True
        all_passed &= complete
        status = "PASS" if complete else "FAIL"
        print(
            f"[{status}] complete: {summary.get('questions_completed')}/"
            f"{summary.get('questions_total')} questions "
            f"(run stopped early if not equal - see --require-complete)"
        )

    for name, required in numeric_checks:
        actual = summary.get(name)
        passed = (actual or 0) >= required
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: actual={actual} required>={required}")

    if not all_passed:
        print("\nOne or more checks failed - failing the gate.")
        return 1

    print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
