"""State management for the indexer graph.

This module defines the state structures and reduction functions used in the
indexer graph.

Classes:
    IndexState: Represents the state for document indexing operations.
    UrlInputState: Represents the input state for web indexing operations.
    WebIndexerState: Represents the state for web indexing operations.

Functions:
    reduce_docs: Processes and reduces document inputs into a sequence of Documents.
    reduce_urls: Processes and reduces url inputs into a sequence of urls.

The module also includes type definitions and utility functions to support
these state management operations.
"""

import logging
from typing import Annotated, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from pydantic import BaseModel

from .._document_indexer.document_indexer_state import reduce_docs

logger = logging.getLogger("states")


def reduce_urls(
    existing: Optional[Sequence[str]],
    new: Union[
        Sequence[str],
        str,
        Literal["delete"],
    ],
) -> Sequence[str]:
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
        return [new]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(item)
        return coerced

    return existing or []


#############################  Web Indexer State  ###################################
class UrlInputState(BaseModel):
    """Input state for web indexer."""

    urls: Annotated[Sequence[str], reduce_urls]


class WebIndexerState(UrlInputState):
    """The State of web indexer."""

    docs: Annotated[Sequence[Document], reduce_docs]
