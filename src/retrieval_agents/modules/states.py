"""States."""

from pydantic import BaseModel, Field


class BasicRAGInputState(BaseModel):
    """Input state for the basic RAG agent."""

    question: str = Field(default="")
