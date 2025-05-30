"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import Annotated, Sequence, cast

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from retrieval_agents import prompts
from retrieval_agents.configurations import IndexerConfiguration
from retrieval_agents.modules import retrieval
from retrieval_agents.modules.utils import (
    format_docs,
    get_message_text,
    load_chat_model,
)


### Schemas ###
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


### Configuration ###
class SimpleRagConfiguration(IndexerConfiguration):
    """The configuration for the agent."""

    response_system_prompt: str = Field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        description="The system prompt used for generating responses.",
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = Field(
        default="openai/gpt-4o",
        description="The language model used for generating responses. Should be in the form: provider/model-name.",
    )

    query_system_prompt: str = Field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        description="The system prompt used for processing and refining queries.",
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = Field(
        default="openai/gpt-4o",
        description="The language model used for processing and refining queries. Should be in the form: provider/model-name.",
    )


### States ###
class SimpleRagInputState(BaseModel):
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


class SimpleRagState(SimpleRagInputState):
    """The state of your graph / agent."""

    queries: Annotated[list[str], add_queries] = Field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = Field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""


### Nodes ###
async def generate_query(
    state: SimpleRagState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = SimpleRagConfiguration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }


async def retrieve(
    state: SimpleRagState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    with retrieval.make_retriever(config) as retriever:
        docs = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": docs}


async def respond(
    state: SimpleRagState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = SimpleRagConfiguration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


### Graph ###
builder = StateGraph(
    SimpleRagState, input=SimpleRagInputState, config_schema=SimpleRagConfiguration
)

builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(respond)
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")
builder.add_edge("respond", END)

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
