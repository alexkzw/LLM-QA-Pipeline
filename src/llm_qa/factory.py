"""Composition root: wire settings, LLM, and pipeline together in one place.

Keeping construction logic here (rather than scattered across scripts) means
every entry point - CLI, API, tests - builds the system the same way.
"""
from __future__ import annotations

from llm_qa.chains.pipeline import QAPipeline
from llm_qa.config.settings import Settings, get_settings
from llm_qa.core.llm_provider import TogetherAILLM
from llm_qa.retrieval.embeddings import EmbeddingModel
from llm_qa.retrieval.retriever import Retriever


def build_pipeline(settings: Settings | None = None) -> QAPipeline:
    """Construct a full-document QAPipeline from settings (no retrieval)."""
    settings = settings or get_settings()
    llm = TogetherAILLM(settings=settings)
    return QAPipeline(llm=llm, settings=settings)


def build_retriever(settings: Settings | None = None) -> Retriever:
    """Construct a Retriever from settings."""
    settings = settings or get_settings()
    return Retriever(
        persist_dir=settings.vector_store_dir,
        embedding_model=EmbeddingModel(settings.embedding_model),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def build_rag_pipeline(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
) -> QAPipeline:
    """Construct a retrieval-augmented QAPipeline."""
    settings = settings or get_settings()
    llm = TogetherAILLM(settings=settings)
    retriever = retriever or build_retriever(settings)
    return QAPipeline(llm=llm, settings=settings, retriever=retriever)
