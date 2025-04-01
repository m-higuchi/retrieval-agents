"""Agent for adaptive RAG."""

import logging
from typing import Dict, Literal, Sequence, cast

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from retrieval_agents.agents import retrieval
from retrieval_agents.agents.configurations import ARagConfiguration
from retrieval_agents.agents.states import AdaptiveRagInputState, ARagState
from retrieval_agents.agents.utils import load_chat_model

logger = logging.getLogger("adaptive_rag")


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


### Nodes ###
async def retrieve(
    state: ARagState, *, config: RunnableConfig
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


async def grade_documents(
    state: ARagState, *, config: RunnableConfig
) -> dict[str, str | Sequence[Document]]:
    """Determine whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    configuration = ARagConfiguration.from_runnable_config(config)
    question = state.question
    documents = state.documents

    llm = load_chat_model(configuration.grade_documents_model)
    structured_llm_grader = llm.with_structured_output(
        GradeDocuments.model_json_schema()
    )

    # Prompt
    system = configuration.grade_documents_system_prompt
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", configuration.grade_documents_human_prompt),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = cast(
            Dict[str, str],
            await retrieval_grader.ainvoke(
                {"question": question, "document": d.page_content}
            ),
        )
        grade = score["binary_score"]
        if grade == "yes":
            logger.info("GRADE: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            logger.info("GRADE: DOCUMENT NOT RELEVANT")
            continue
    return {"documents": filtered_docs, "question": question}


async def generate(
    state: ARagState, *, config: RunnableConfig
) -> dict[str, str | int | Sequence[Document]]:
    """Generate answer.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    configuration = ARagConfiguration.from_runnable_config(config)

    question = state.question
    documents = state.documents
    prompt = ChatPromptTemplate.from_messages(
        [("human", configuration.generate_human_prompt)]
    )
    llm = load_chat_model(configuration.generate_model)
    rag_chain = prompt | llm | StrOutputParser()
    # RAG generation
    generation = await rag_chain.ainvoke({"context": documents, "question": question})

    state.generation = generation
    state.generation_count = state.generation_count + 1
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_count": state.generation_count + 1,
    }


async def web_search(
    state: ARagState, *, config: RunnableConfig
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
    state: ARagState, *, config: RunnableConfig
) -> dict[str, str | Sequence[Document]]:
    """Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    question = state.question
    documents = state.documents
    configuration = ARagConfiguration.from_runnable_config(config)
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


async def route_question(state: ARagState, *, config: RunnableConfig) -> str:
    """Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    configuration = ARagConfiguration.from_runnable_config(config)
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


def decide_to_generate(state: ARagState) -> str:
    """Determine whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    filtered_documents = state.documents

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info(
            "DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        logger.info("DECISION: GENERATE")
        return "generate"


async def grade_generation_v_documents_and_question(
    state: ARagState, *, config: RunnableConfig
) -> str:
    """Determine whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    configuration = ARagConfiguration.from_runnable_config(config)
    grade_hallucination = (
        await _grade_generation_v_documents_and_question_hallucination(
            state=state, configuration=configuration
        )
    )

    # Check hallucination
    if grade_hallucination:
        grade_answer = await _grade_generation_v_docuemnts_and_question_answer(
            state=state, configuration=configuration
        )
        if not grade_answer:
            logger.info("DECISION: GENERATION DOES NOT ADDRESS QUESTION")
            if state.generation_count > configuration.max_generation:
                logger.info("DECISION: REACHED MAX GENERATION COUNT")
                return "end"
            else:
                return "not useful"
        else:
            logger.info("DECISION: GENERATION ADDRESSES QUESTION")
            return "useful"
    elif state.generation_count > configuration.max_generation:
        logger.info("DECISION: REACHED MAX GENERATION COUNT")
        return "end"
    else:
        logger.info("DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY")
        return "not supported"


async def _grade_generation_v_documents_and_question_hallucination(
    state: ARagState, configuration: ARagConfiguration
) -> bool:
    documents = state.documents
    generation = state.generation

    llm = load_chat_model(configuration.hallucination_grader_model)
    configuration.hallucination_grader_human_prompt
    structured_llm_grader = llm.with_structured_output(
        GradeHallucinations.model_json_schema(), include_raw=True
    )
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.hallucination_grader_system_prompt),
            ("human", configuration.hallucination_grader_human_prompt),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_grader
    response = cast(
        Dict[str, Dict[str, str]],
        await hallucination_grader.ainvoke(
            {"documents": documents, "generation": generation}
        ),
    )
    # _ = response["raw"]
    score = response["parsed"]
    # _ = response["parsing_error"]

    grade = score["binary_score"]
    if grade == "yes":
        return True
    else:
        return False


async def _grade_generation_v_docuemnts_and_question_answer(
    state: ARagState, configuration: ARagConfiguration
) -> bool:
    question = state.question
    generation = state.generation
    # Check question-answering
    llm = load_chat_model(configuration.answer_grader_model)
    structured_llm_grader = llm.with_structured_output(GradeAnswer.model_json_schema())
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.answer_grader_system_prompt),
            ("human", configuration.answer_grader_human_prompt),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    score = cast(
        Dict[str, str],
        await answer_grader.ainvoke({"question": question, "generation": generation}),
    )
    grade = score["binary_score"]
    if grade == "yes":
        return True
    else:
        return False


### Build Graph ###
graph_name = "AdaptiveRAGGaph"
logger.info(f"Building {graph_name}")
builder = StateGraph(
    ARagState, input=AdaptiveRagInputState, config_schema=ARagConfiguration
)

builder.add_node(web_search)
builder.add_node(retrieve)
builder.add_node(grade_documents)
builder.add_node(generate)
builder.add_node(transform_query)

builder.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
builder.add_edge("web_search", "generate")
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
builder.add_edge("transform_query", "retrieve")
builder.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not supported": "generate", "useful": END, "not useful": "transform_query"},
)

graph: CompiledStateGraph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = graph_name
