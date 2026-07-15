"""Shared groundedness check, used by both single-model and ensemble validation."""
from __future__ import annotations


def is_fully_grounded(validation_text: str) -> bool:
    """Return True only if the validator flagged no unsupported claims.

    Normalises case and looks for the specific negative labels as whole
    tokens, rather than a raw substring check.
    """
    upper = validation_text.upper()
    return "UNSUPPORTED" not in upper and "PARTIALLY SUPPORTED" not in upper
