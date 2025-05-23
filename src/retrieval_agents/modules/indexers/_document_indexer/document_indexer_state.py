"""State management for the indexer graph.

This module defines the state structures and reduction functions used in the
indexer graph.

Classes:
    DocumentIndexer: Represents the state for document indexing operations.

Functions:
    reduce_docs: Processes and reduces document inputs into a sequence of Documents.

The module also includes type definitions and utility functions to support
these state management operations.
"""

import logging
import uuid
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger("states")
############################  Doc Indexer State  #############################


def reduce_docs(
    existing: Optional[Sequence[Document]],
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            else:
                coerced.append(item)
        return coerced
    return existing or []


# The index state defines the simple IO for the single-node index graph
class DocumentIndexerState(BaseModel):
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """

    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""
