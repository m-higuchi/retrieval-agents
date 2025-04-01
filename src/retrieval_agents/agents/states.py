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
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from retrieval_agents.indexers.states import reduce_docs

logger = logging.getLogger("states")

############################# Simple RAG Agent State  ###################################


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
class InputState(BaseModel):
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


# This is the primary state of your agent, where you can store any information


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)


class State(InputState):
    """The state of your graph / agent."""

    queries: Annotated[list[str], add_queries] = Field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = Field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""


#############################  Adaptive RAG Agent State  ###################################


class AdaptiveRagInputState(BaseModel):
    """Input state for adaptive rag."""

    question: str = Field()


class ARagState(AdaptiveRagInputState):
    """The state of the adaptive RAG agent."""

    documents: Annotated[Sequence[Document], reduce_docs]
    generation: str = Field(default="")
    generation_count: int = Field(default=0)
