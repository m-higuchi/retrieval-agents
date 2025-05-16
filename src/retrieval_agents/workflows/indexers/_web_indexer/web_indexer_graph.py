"""This "graph" exposes an endpoint for a user to upload URLs to be indexed."""

import logging
from typing import Optional, Sequence

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from retrieval_agents.workflows import retrieval
from retrieval_agents.workflows.indexers._web_indexer.web_indexer_state import (
    UrlInputState,
    WebIndexerState,
)
from retrieval_agents.workflows.indexers.configurations import IndexerConfiguration

logger = logging.getLogger("web_indexer")


def ensure_docs_have_user_id(
    docs: Sequence[Document], config: RunnableConfig
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        docs (Sequence[Document]): A sequence of Document objects to process.
        config (RunnableConfig): A configuration object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    configurable = config.get("configurable") or {}
    user_id = configurable["user_id"]
    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in docs
    ]


async def load_web(
    state: WebIndexerState, *, config: Optional[RunnableConfig] = None
) -> dict[str, list[Document]]:
    """Load from the web sites.

    Args:
        state (WebIndexState): Input state.
        config (Optional[RunnableConfig], optional): Runnable config. Defaults to None.
    """
    urls = [url for url in state.urls]
    loader = WebBaseLoader(urls)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    # state.docs = [item for sublist in docs for item in sublist]
    return {"docs": docs}


async def split_text(
    state: WebIndexerState, *, config: Optional[RunnableConfig] = None
) -> WebIndexerState:
    """Split documents.

    Args:
        state (WebIndexState): Input state.
        config (Optional[RunnableConfig], optional): Runnable config. Defaults to None.

    Returns:
        WebIndexState: The updated state after splitting the text in the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    docs = text_splitter.split_documents(state.docs)
    state.docs = docs
    return state


async def index_docs(
    state: WebIndexerState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (WebIndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    logger.debug(state)
    if not config:
        raise ValueError("Configuration required to run index_docs.")
    with retrieval.make_retriever(config) as retriever:
        stamped_docs = ensure_docs_have_user_id(state.docs, config)

        await retriever.aadd_documents(stamped_docs)
    return {"docs": "delete"}


# Define a new graph


builder = StateGraph(
    WebIndexerState, input=UrlInputState, config_schema=IndexerConfiguration
)
builder.add_node(index_docs)
builder.add_node(load_web)
builder.add_node(split_text)
builder.add_edge(START, "load_web")
builder.add_edge("load_web", "split_text")
builder.add_edge("split_text", "index_docs")
builder.add_edge("index_docs", END)
# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "WebIndex"
