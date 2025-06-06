"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Annotated, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from retrieval_agents.configurations import IndexerConfiguration
from retrieval_agents.modules.retrieval import make_retriever
from retrieval_agents.modules.utils import reduce_docs


### States ###
class DocumentIndexerState(BaseModel):
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """

    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


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


### Nodes ###
async def index_docs(
    state: DocumentIndexerState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    with make_retriever(config) as retriever:
        stamped_docs = ensure_docs_have_user_id(state.docs, config)

        await retriever.aadd_documents(stamped_docs)
    return {"docs": "delete"}


### Graph ###
builder = StateGraph(DocumentIndexerState, config_schema=IndexerConfiguration)
builder.add_node(index_docs)
builder.set_entry_point("index_docs")
builder.set_finish_point("index_docs")

graph = builder.compile()
graph.name = "DocumentIndexer"
