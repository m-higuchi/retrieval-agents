"""Define the configurable parameters for the agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field


class ConfigurationBase(BaseModel):
    """Base class for configuration."""

    user_id: str = Field(description="Unique identifier for the user.")

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create a BaseConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of BaseConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = set(cls.model_fields.keys())
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


class IndexerConfiguration(ConfigurationBase):
    """Configuration form indexers."""

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = Field(
        default="openai/text-embedding-3-small",
        description="Name of the embedding model to use. Must be a valid embedding model name.",
    )

    index_name: str | None = Field(
        default=None, description="The name of index to retrieve."
    )
    retriever_provider: Annotated[
        Literal["elastic", "elastic-local", "pinecone", "mongodb", "chroma"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = Field(
        default="elastic",
        description="The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', 'mongodb' or 'chroma'.",
    )

    search_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the search function of the retriever.",
    )


T = TypeVar("T", bound=ConfigurationBase)
