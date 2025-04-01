"""Retrieval Graph Module

This module provides a conversational retrieval graph system that enables
intelligent document retrieval and question answering based on user inputs.

The main components of this system include:

1. A state management system for handling conversation context and document retrieval.
2. A query generation mechanism that refines user inputs into effective search queries.
3. A document retrieval system that fetches relevant information based on generated queries.
4. A response generation system that formulates answers using retrieved documents and conversation history.

The graph is configured using customizable parameters defined in the Configuration class,
allowing for flexibility in model selection, retrieval methods, and system prompts.

Key Features:
- Adaptive query generation for improved document retrieval
- Integration with various retrieval providers (e.g., Elastic, Pinecone, MongoDB)
- Customizable language models for query and response generation
- Stateful conversation management for context-aware interactions

Usage:
    The main entry point for using this system is the `graph` object exported from this module.
    It can be invoked to process user inputs and generate responses based on retrieved information.

For detailed configuration options and usage instructions, refer to the Configuration class
and individual component documentation within the retrieval_graph package.
"""  # noqa

from retrieval_agents.agents.simple_rag import graph as simple_rag
from retrieval_agents.indexers.document_indexer import graph as indexer_graph
from retrieval_agents.indexers.web_indexer import graph as web_indexer_graph

__all__ = ["simple_rag", "indexer_graph", "web_indexer_graph"]
