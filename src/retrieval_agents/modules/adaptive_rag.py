"""Agent for adaptive RAG."""

import logging
from typing import Annotated, Dict, Literal, Sequence, cast

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from retrieval_agents import prompts
from retrieval_agents.configurations import IndexerConfiguration
from retrieval_agents.modules import retrieval
from retrieval_agents.modules.contextual_answer_generator import (
    ContextualAnswerGeneratorConfiguration,
    ContextualAnswerGeneratorState,
)
from retrieval_agents.modules.contextual_answer_generator import (
    graph as retrieval_generator_graph,
)
from retrieval_agents.modules.states import BasicRAGInputState
from retrieval_agents.modules.utils import load_chat_model

logger = logging.getLogger("adaptive_rag_graph2")


### Configuration ###
class AdaptiveRagConfiguration(
    IndexerConfiguration, ContextualAnswerGeneratorConfiguration
):
    """The configuration for the adaptive rag agent."""

    router_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = Field(
        default="openai/gpt-4o",
        description="The language model used for routing the user questions.",
    )

    router_system_prompt: str = Field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        description="The system prompt used for routing questions.",
    )

    topics: str = Field(
        default="agents, prompt engineering, and adversarial attacks",
        description="The topics to retrieve.",
    )
    rewrite_system_prompt: str = Field(
        default=prompts.REWRITE_SYSTEM_PROMPT,
        description="The prompt used for rewrite the question.",
    )

    rewrite_human_propmt: str = Field(
        default=prompts.REWRITE_HUMAN_PROMPT,
    )

    rewrite_model: Annotated[str, {"__metadata__": {"kind", "llm"}}] = Field(
        default="openai/gpt-4o",
        description="The language model used for rewriting the questions.",
    )


### Schemas ###
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


### States ###


### Nodes ###
async def retrieve(
    state: BasicRAGInputState, *, config: RunnableConfig
) -> dict[str, str | Sequence[Document]]:
    """Retrieve documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # configuration = ARagConfiguration.from_runnable_config(config)
    question = state.question

    # Retrieval
    # if retriever:
    #    state.documents = await retriever.ainvoke(question, config)
    with retrieval.make_retriever(config=config) as retriever:
        documents = await retriever.ainvoke(question, config)
    return {"question": question, "documents": documents}


async def web_search(
    state: BasicRAGInputState, *, config: RunnableConfig
) -> dict[str, str | Sequence[Document]]:
    """Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state.question

    # Web search
    from langchain_community.tools.tavily_search import TavilySearchResults

    web_search_tool = TavilySearchResults(k=3)
    docs = await web_search_tool.ainvoke({"query": question})
    web_results = [Document(page_content="\n".join([d["content"] for d in docs]))]

    return {"documents": web_results, "question": question}


async def transform_query(
    state: ContextualAnswerGeneratorState, *, config: RunnableConfig
) -> dict[str, str | Sequence[Document]]:
    """Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    question = state.question
    documents = state.documents
    configuration = AdaptiveRagConfiguration.from_runnable_config(config)
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.rewrite_system_prompt),
            ("human", configuration.rewrite_human_propmt),
        ]
    )

    llm = load_chat_model(configuration.rewrite_model)
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    # Re-write question
    better_question = await question_rewriter.ainvoke({"question": question})
    return {"documents": documents, "question": better_question}


### Edges ###


async def route_question(state: BasicRAGInputState, *, config: RunnableConfig) -> str:
    """Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    configuration = AdaptiveRagConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.router_model)
    structured_llm_router = llm.with_structured_output(RouteQuery.model_json_schema())
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.router_system_prompt),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    question = state.question
    source = cast(
        Dict[str, str],
        await question_router.ainvoke(
            {"question": question, "topics": configuration.topics}
        ),
    )
    if source["datasource"] == "web_search":
        return "web_search"
    elif source["datasource"] == "vectorstore":
        return "vectorstore"
    else:
        return "web_search"


### Build Graph ###
graph_name = "AdaptiveRAGGaph2"
logger.info(f"Building {graph_name}")
builder = StateGraph(
    ContextualAnswerGeneratorState,
    input=BasicRAGInputState,
    config_schema=AdaptiveRagConfiguration,
)

builder.add_node(web_search)
builder.add_node(retrieve)
builder.add_node("retrieval_generator_graph", retrieval_generator_graph)
builder.add_node(transform_query)

builder.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)


def _transform_query_or_end(state: ContextualAnswerGeneratorState) -> str:
    logger.info(state)
    if state.finish_reason == "complete":
        return "complete"
    else:
        return "transform_query"


builder.add_edge("web_search", "retrieval_generator_graph")
builder.add_edge("retrieve", "retrieval_generator_graph")
builder.add_conditional_edges(
    "retrieval_generator_graph",
    _transform_query_or_end,
    {
        "complete": END,
        "transform_query": "transform_query",
    },
)
builder.add_edge("transform_query", "retrieve")

graph: CompiledStateGraph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = graph_name

__all__ = ["ContextualAnswerGeneratorState"]
