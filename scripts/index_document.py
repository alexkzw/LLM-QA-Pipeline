#!/usr/bin/env python3
"""Standalone CLI: index a PDF into the vector store for RAG.

One job: read a PDF, chunk it, embed the chunks, and persist them to the Chroma
index so the RAG pipeline can retrieve from them.

Safe to run every time, on any PDF - you don't need to remember whether a
given document was already indexed. A fingerprint of the document's content,
chunk settings, and embedding model is persisted alongside the index:
  * same document, same settings  -> detected automatically, skipped (no-op)
  * different document or settings -> detected automatically, rebuilt
  * --force                        -> always rebuilds, regardless

Example:
    python scripts/index_document.py --pdf data/oecd_economic_outlook_2026.pdf
    python scripts/index_document.py --pdf data/oecd_economic_outlook_2026.pdf --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_qa.config.settings import get_settings
from llm_qa.core.document_loader import load_pdf_text
from llm_qa.core.exceptions import LLMQAError
from llm_qa.core.logging_config import configure_logging, get_logger
from llm_qa.factory import build_retriever

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path, help="PDF to index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if it already contains chunks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    try:
        # Large documents: raise the char ceiling well above the default,
        # since retrieval (not prompt-stuffing) handles the size.
        text = load_pdf_text(args.pdf, max_chars=50_000_000)
        retriever = build_retriever(settings)
        count = retriever.index_document(text, force=args.force)
    except LLMQAError as exc:
        logger.error("Indexing failed: %s", exc)
        return 1

    logger.info(
        "Done. Index contains %d chunks at '%s'.",
        count,
        settings.vector_store_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
