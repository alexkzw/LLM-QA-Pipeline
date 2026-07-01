"""Reference-document loading and validation.

Refactored from the notebook's ``get_pdf_text``: adds path validation, typed
errors, a configurable size limit, and a clean separation between "read the
file" and "validate the content".
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from llm_qa.core.exceptions import DocumentError, DocumentTooLargeError
from llm_qa.core.logging_config import get_logger

logger = get_logger(__name__)


def load_pdf_text(file_path: str | Path, max_chars: int) -> str:
    """Extract and validate text from a PDF reference document.

    Args:
        file_path: Path to the PDF on disk.
        max_chars: Maximum allowed character count for the extracted text.

    Returns:
        The concatenated text content of the PDF.

    Raises:
        DocumentError: If the file is missing or cannot be parsed.
        DocumentTooLargeError: If the extracted text exceeds ``max_chars``.
    """
    path = Path(file_path)

    if not path.exists():
        raise DocumentError(f"Reference document not found: {path}")
    if not path.is_file():
        raise DocumentError(f"Reference path is not a file: {path}")

    logger.info("Reading reference document: %s", path)

    try:
        with path.open("rb") as handle:
            reader = PdfReader(handle)
            pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
        raise DocumentError(f"Failed to parse PDF '{path}': {exc}") from exc

    text = "\n\n".join(pages).strip()

    if not text:
        raise DocumentError(f"No extractable text found in '{path}'.")

    if len(text) > max_chars:
        raise DocumentTooLargeError(
            f"Reference document has {len(text):,} characters; "
            f"limit is {max_chars:,}."
        )

    logger.info("Loaded reference document (%d characters).", len(text))
    return text
