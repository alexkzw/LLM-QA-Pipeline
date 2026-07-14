"""Document chunking for retrieval.

Splits a long document into overlapping, sentence-aware chunks small enough to
embed and retrieve precisely. Overlap preserves context that would otherwise be
severed at a chunk boundary.

This is the step that makes a 300-page document tractable: instead of stuffing
the whole document into every prompt, we index small chunks and retrieve only
the few that are relevant to each question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from llm_qa.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A retrievable unit of text with provenance metadata."""

    chunk_id: int
    text: str
    char_start: int
    char_end: int


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no heavy NLP dependency).

    Splits on sentence-ending punctuation followed by whitespace. Good enough
    for chunking prose; we are not doing linguistic analysis here.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` chars.

    Args:
        text: The full document text.
        chunk_size: Target maximum characters per chunk.    
        chunk_overlap: Characters of trailing context to repeat at the start of
            the next chunk, so facts spanning a boundary are not lost.

    Returns:
        A list of Chunk objects with character offsets for provenance.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    sentences = _split_sentences(text)
    chunks: list[Chunk] = [] # the growing output list
    buffer = "" # the text of chunk currently being built
    buffer_start = 0 #
    cursor = 0  # running char position in the original text

    # take whatever's currently in the buffer and turn it into a real 
    # Chunk appended to the output list
    def _flush(buf: str, start: int) -> None:
        # guards against creating an empty chunk
        if buf.strip():
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=buf.strip(),
                    char_start=start,
                    char_end=start + len(buf),
                )
            )

    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > chunk_size:
            _flush(buffer, buffer_start)
            # Start the next buffer with the overlap tail of the previous one.
            tail = buffer[-chunk_overlap:] if chunk_overlap else ""
            buffer_start = cursor - len(tail)
            buffer = (tail + " " + sentence).strip()
        else:
            if not buffer:
                buffer_start = cursor
            buffer = (buffer + " " + sentence).strip() if buffer else sentence
        cursor += len(sentence) + 1

    _flush(buffer, buffer_start)

    logger.info(
        "Chunked %d chars into %d chunks (size=%d, overlap=%d).",
        len(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
