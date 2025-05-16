"""State management for the retrieval graph.

This module defines the state structures and reduction functions used in the
retrieval graph. It includes definitions for document indexing, retrieval,
and conversation management.

Classes:
    IndexState: Represents the state for document indexing operations.
    RetrievalState: Represents the state for document retrieval operations.
    ConversationState: Represents the state of the ongoing conversation.

Functions:
    reduce_docs: Processes and reduces document inputs into a sequence of Documents.
    reduce_retriever: Updates the retriever in the state.
    reduce_messages: Manages the addition of new messages to the conversation state.
    reduce_retrieved_docs: Handles the updating of retrieved documents in the state.

The module also includes type definitions and utility functions to support
these state management operations.
"""

import logging
from typing import Annotated, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from retrieval_agents.workflows.indexers._document_indexer.document_indexer_state import (
    reduce_docs,
)

logger = logging.getLogger("adaptive_rag_state")


#############################  Adaptive RAG Agent State  ###################################
class AdaptiveRagInputState(BaseModel):
    """Input state for adaptive rag."""

    question: str = Field()


class AdaptiveRagState(AdaptiveRagInputState):
    """The state of the adaptive RAG agent."""

    documents: Annotated[Sequence[Document], reduce_docs]
    generation: str = Field(default="")
    generation_count: int = Field(default=0)
