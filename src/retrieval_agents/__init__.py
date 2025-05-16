"""Retrieval Agents package."""

from langchain_core.runnables import RunnableConfig

from .workflows import (
    AdaptiveRagConfiguration,
    AdaptiveRagInputState,
    DocumentIndexerState,
    IndexerConfiguration,
    SimpleRagConfiguration,
    SimpleRagInputState,
    UrlInputState,
    adaptive_rag,
    document_indexer,
    simple_rag,
    web_indexer,
)

__all__ = [
    "adaptive_rag",
    "document_indexer",
    "simple_rag",
    "SimpleRagConfiguration",
    "SimpleRagInputState",
    "AdaptiveRagInputState",
    "AdaptiveRagConfiguration",
    "RunnableConfig",
    "indexer",
    "UrlInputState",
    "web_indexer",
    "DocumentIndexerState",
    "IndexerConfiguration",
]

from .logging_config import setup_logging

setup_logging()
