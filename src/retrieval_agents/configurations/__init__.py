"""Alias for configurations."""

from retrieval_agents.modules import IndexerConfiguration
from retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_configuration import (
    AdaptiveRagConfiguration,
)
from retrieval_agents.modules.rag._simple_rag.simple_rag_configuration import (
    SimpleRagConfiguration,
)

__all__ = ["AdaptiveRagConfiguration", "SimpleRagConfiguration", "IndexerConfiguration"]
