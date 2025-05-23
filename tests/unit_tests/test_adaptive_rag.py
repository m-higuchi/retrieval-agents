from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from pytest import fixture, mark

from retrieval_agents import RunnableConfig
from retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph import (
    _grade_generation_v_docuemnts_and_question_answer,
    _grade_generation_v_documents_and_question_hallucination,
    decide_to_generate,
    generate,
    grade_documents,
    grade_generation_v_documents_and_question,
    retrieve,
    route_question,
    transform_query,
    web_search,
)
from retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_state import (
    AdaptiveRagState,
)


@fixture
def runnable_config() -> dict[str, dict[str, str]]:
    # configurable = ARagConfiguration(user_id="test_user")
    # return RunnableConfig(configurable=configurable)
    return {
        "configurable": {
            "user_id": "test_user",
        }
    }


@pytest.fixture
def mock_chat_model() -> MagicMock:
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = "mocked structured output"
    return mock_model


### Nodes ###
@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.retrieval.make_retriever"
)
async def test_retrieve(
    mock_make_retriever: MagicMock, runnable_config: RunnableConfig
) -> None:
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value="retrieved docs")

    mock_make_retriever.return_value.__enter__.return_value = mock_retriever

    state = AdaptiveRagState(question="agent memory", documents=[])
    result = await retrieve(state=state, config=runnable_config)
    assert result["documents"] == "retrieved docs"


@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.ChatPromptTemplate"
)
@patch("retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.load_chat_model")
async def test_grade_documents(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    runnable_config: RunnableConfig,
) -> None:
    mock_response = MagicMock()
    mock_response.binary_score = "yes"

    async def response(arg: dict[str, str]) -> dict[str, str]:
        if arg["document"] == "relevant":
            return {"binary_score": "yes"}
        else:
            return {"binary_score": "no"}

    mock_retrieval_grader = MagicMock()
    mock_retrieval_grader.ainvoke = AsyncMock(side_effect=response)

    # mock_structured_llm = MagicMock()

    mock_llm = MagicMock()
    # mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_load_chat_model.return_value = mock_llm

    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_retrieval_grader
    mock_prompt_cls.from_messages.return_value = mock_prompt

    state = AdaptiveRagState(
        question="agent memory",
        documents=[
            Document(page_content="relevant"),
            Document(page_content="irrelevant"),
        ],
    )
    result = await grade_documents(state=state, config=runnable_config)
    assert result["question"] == state.question
    assert result["documents"] == [Document(page_content="relevant")]


@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.ChatPromptTemplate"
)
@patch("retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.load_chat_model")
async def test_generate(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    runnable_config: RunnableConfig,
) -> None:
    mock_load_chat_model.return_value = MagicMock()

    mock_rag_chain = MagicMock()
    mock_rag_chain.ainvoke = AsyncMock(return_value="generated text")

    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value.__or__.return_value = mock_rag_chain
    mock_prompt_cls.from_messages.return_value = mock_prompt
    response = await generate(
        state=AdaptiveRagState(question="agent memory", documents=[]),
        config=runnable_config,
    )
    assert response["documents"] == []
    assert response["question"] == "agent memory"
    assert response["generation"] == "generated text"


@mark.asyncio
@patch("langchain_community.tools.tavily_search.TavilySearchResults")
async def test_web_search(
    mock_tavily_search_results: MagicMock, runnable_config: RunnableConfig
) -> None:
    mock_web_search_tool = MagicMock()
    mock_web_search_tool.ainvoke = AsyncMock(
        return_value=[{"content": "searched doc1"}, {"content": "searched doc2"}]
    )

    mock_tavily_search_results.return_value = mock_web_search_tool

    response = await web_search(
        state=AdaptiveRagState(question="agent memory", documents=[]),
        config=runnable_config,
    )
    assert response["documents"] == [
        Document(page_content="searched doc1\nsearched doc2")
    ]
    assert response["question"] == "agent memory"


@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.ChatPromptTemplate"
)
@patch("retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.load_chat_model")
async def test_transform_query(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    runnable_config: RunnableConfig,
) -> None:
    mock_load_chat_model.return_value = MagicMock()

    mock_question_rewrite = MagicMock()
    mock_question_rewrite.ainvoke = AsyncMock(return_value="better question")

    mock_re_write_prompt = MagicMock()
    mock_re_write_prompt.__or__.return_value.__or__.return_value = mock_question_rewrite

    mock_prompt_cls.from_messages.return_value = mock_re_write_prompt
    response = await transform_query(
        state=AdaptiveRagState(question="agent memory", documents=[]),
        config=runnable_config,
    )
    assert response["documents"] == []
    assert response["question"] == "better question"


### Edges ###
@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.ChatPromptTemplate"
)
@patch("retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.load_chat_model")
@mark.parametrize(
    "relevance, expected",
    [("vectorstore", "vectorstore"), ("web_search", "web_search")],
)
async def test_route_question_rag(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    relevance: str,
    expected: str,
) -> None:
    mock_question_router = MagicMock()
    mock_question_router.ainvoke = AsyncMock(return_value={"datasource": relevance})

    # mock_structured_llm = MagicMock()
    # mock_structured_llm.__or__.return_value = mock_question_router

    mock_llm = MagicMock()
    # mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_load_chat_model.return_value = mock_llm
    # mock_load_chat_model.return_value = None

    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_question_router
    mock_prompt_cls.from_messages.return_value = mock_prompt

    result = await route_question(
        state=AdaptiveRagState(question="agent memory", documents=[]),
        config={"configurable": {"user_id": "test_usser", "topics": "test topics"}},
    )
    assert result == expected
    mock_question_router.ainvoke.assert_awaited_once_with(
        {"question": "agent memory", "topics": "test topics"}
    )


@mark.asyncio
@mark.parametrize(
    "documents, expected",
    [([], "transform_query"), ([Document(page_content="")], "generate")],
)
async def test_decide_to_generate(documents: list[Document], expected: str) -> None:
    result = decide_to_generate(
        state=AdaptiveRagState(question="agent memory", documents=documents)
    )
    assert result == expected


@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph._grade_generation_v_documents_and_question_hallucination",
    new_callable=AsyncMock,
)
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph._grade_generation_v_docuemnts_and_question_answer",
    new_callable=AsyncMock,
)
@mark.parametrize(
    "hallucination_grade, answer_grade, generation_count, expected",
    [
        (True, True, 3, "useful"),
        (True, True, 4, "useful"),
        (True, False, 3, "not useful"),
        (True, False, 4, "end"),
        (False, True, 3, "not supported"),
        (False, True, 4, "end"),
        (False, False, 3, "not supported"),
        (False, False, 4, "end"),
    ],
)
async def test_grade_generation_v_documents_and_question(
    mock_grade_generation_v_docuemnts_and_question_answer: MagicMock,
    mock_grade_generation_v_documents_and_question_hallucination: MagicMock,
    runnable_config: RunnableConfig,
    hallucination_grade: bool,
    answer_grade: bool,
    generation_count: int,
    expected: str,
) -> None:
    runnable_config["configurable"]["max_generation"] = 3
    mock_grade_generation_v_docuemnts_and_question_answer.return_value = answer_grade
    mock_grade_generation_v_documents_and_question_hallucination.return_value = (
        hallucination_grade
    )
    response = await grade_generation_v_documents_and_question(
        state=AdaptiveRagState(
            question="", documents=[], generation_count=generation_count
        ),
        config=runnable_config,
    )
    assert response == expected


@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.ChatPromptTemplate"
)
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.load_chat_model",
    return_value=MagicMock(),
)
@mark.parametrize("binary_score, expected", [("yes", True), ("No", False)])
async def test_grade_generation_v_documents_and_question_hallucination(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    binary_score: str,
    expected: bool,
) -> None:
    mock_llm = MagicMock()
    mock_structured_llm_grader = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm_grader
    mock_load_chat_model.return_value = mock_llm

    mock_hallucination_prompt = MagicMock()
    mock_prompt_cls.from_messages.return_value = mock_hallucination_prompt

    hallucination_grader = MagicMock()
    mock_hallucination_prompt.__or__.return_value = hallucination_grader

    hallucination_grader.ainvoke = AsyncMock(
        return_value={
            "raw": "raw_expected",
            "parsed": {"binary_score": binary_score},
            "parsing_error": "parsing_error_expected",
        }
    )
    response = await _grade_generation_v_documents_and_question_hallucination(
        state=AdaptiveRagState(question="", documents=[]), configuration=MagicMock()
    )
    assert response == expected


@mark.asyncio
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.ChatPromptTemplate"
)
@patch(
    "retrieval_agents.modules.rag._adaptive_rag.adaptive_rag_graph.load_chat_model",
    return_value=MagicMock(),
)
@mark.parametrize("binary_score, expected", [("yes", True), ("no", False)])
async def test_grade_generation_v_docuemnts_and_question_answer(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    binary_score: str,
    expected: bool,
) -> None:
    mock_llm = MagicMock()
    mock_structured_llm_grader = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm_grader
    mock_load_chat_model.return_value = mock_llm

    mock_answer_prompt = MagicMock()

    answer_grader = MagicMock()
    mock_answer_prompt.__or__.return_value = answer_grader

    mock_prompt_cls.from_messages.return_value = mock_answer_prompt
    answer_grader.ainvoke = AsyncMock(return_value={"binary_score": binary_score})
    response = await _grade_generation_v_docuemnts_and_question_answer(
        state=AdaptiveRagState(question="", documents=[]), configuration=MagicMock()
    )
    assert response == expected
