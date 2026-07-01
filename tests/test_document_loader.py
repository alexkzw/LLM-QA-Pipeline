"""Tests for the document loader's validation logic."""
from __future__ import annotations

import pytest

from llm_qa.core.document_loader import load_pdf_text
from llm_qa.core.exceptions import DocumentError


def test_missing_file_raises_document_error(tmp_path) -> None:
    missing = tmp_path / "nope.pdf"
    with pytest.raises(DocumentError, match="not found"):
        load_pdf_text(missing, max_chars=1000)


def test_directory_path_raises_document_error(tmp_path) -> None:
    with pytest.raises(DocumentError, match="not a file"):
        load_pdf_text(tmp_path, max_chars=1000)
