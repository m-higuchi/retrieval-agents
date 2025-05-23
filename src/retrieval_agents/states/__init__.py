"""Alias for states."""

from retrieval_agents.modules.indexers._document_indexer.document_indexer_state import (
    DocumentIndexerState,
)
from retrieval_agents.modules.indexers._web_indexer.web_indexer_state import (
    UrlInputState,
    WebIndexerState,
)
from retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_state import (
    AdaptiveRagInputState,
    AdaptiveRagState,
)
from retrieval_agents.modules.rag._simple_rag.simple_rag_state import (
    SimpleRagInputState,
    SimpleRagState,
)

__all__ = [
    "AdaptiveRagState",
    "AdaptiveRagInputState",
    "SimpleRagState",
    "SimpleRagInputState",
    "DocumentIndexerState",
    "UrlInputState",
    "WebIndexerState",
]
