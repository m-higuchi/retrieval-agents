from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from pytest import fixture, mark

from retrieval_agents import RunnableConfig
from retrieval_agents.modules.adaptive_rag import (
    ContextualAnswerGeneratorState,
    retrieve,
    route_question,
    transform_query,
    web_search,
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
@patch("retrieval_agents.modules.adaptive_rag.retrieval.make_retriever")
async def test_retrieve(
    mock_make_retriever: MagicMock, runnable_config: RunnableConfig
) -> None:
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value="retrieved docs")

    mock_make_retriever.return_value.__enter__.return_value = mock_retriever

    state = ContextualAnswerGeneratorState(question="agent memory", documents=[])
    result = await retrieve(state=state, config=runnable_config)
    assert result["documents"] == "retrieved docs"


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
        state=ContextualAnswerGeneratorState(question="agent memory", documents=[]),
        config=runnable_config,
    )
    assert response["documents"] == [
        Document(page_content="searched doc1\nsearched doc2")
    ]
    assert response["question"] == "agent memory"


@mark.asyncio
@patch("retrieval_agents.modules.adaptive_rag.ChatPromptTemplate")
@patch("retrieval_agents.modules.adaptive_rag.load_chat_model")
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
        state=ContextualAnswerGeneratorState(question="agent memory", documents=[]),
        config=runnable_config,
    )
    assert response["documents"] == []
    assert response["question"] == "better question"


@mark.asyncio
@patch("retrieval_agents.modules.adaptive_rag.ChatPromptTemplate")
@patch("retrieval_agents.modules.adaptive_rag.load_chat_model")
@mark.parametrize(
    "relevance, expected",
    [("vectorstore", "vectorstore"), ("web_search", "web_search")],
)
async def test_route_question(
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
        state=ContextualAnswerGeneratorState(question="agent memory", documents=[]),
        config={"configurable": {"user_id": "test_usser", "topics": "test topics"}},
    )
    assert result == expected
    mock_question_router.ainvoke.assert_awaited_once_with(
        {"question": "agent memory", "topics": "test topics"}
    )
