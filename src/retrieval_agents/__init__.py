"""Retrieval Agents package."""

from langchain_core.runnables import RunnableConfig

from .modules import (
    AdaptiveRagConfiguration,
    ContextualAnswerGeneratorConfiguration,
    ContextualAnswerGeneratorInputState,
    DocumentIndexerState,
    IndexerConfiguration,
    SimpleRagConfiguration,
    SimpleRagInputState,
    UrlInputState,
    adaptive_rag,
    contextual_answer_generator,
    document_indexer,
    simple_rag,
    web_indexer,
)

__all__ = [
    "adaptive_rag",
    "contextual_answer_generator",
    "document_indexer",
    "simple_rag",
    "SimpleRagConfiguration",
    "SimpleRagInputState",
    "ContextualAnswerGeneratorConfiguration",
    "ContextualAnswerGeneratorInputState",
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
