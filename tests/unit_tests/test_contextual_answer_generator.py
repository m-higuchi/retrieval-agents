from typing import Sequence, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from pytest import fixture, mark

from retrieval_agents import RunnableConfig
from retrieval_agents.modules.contextual_answer_generator import (
    ContextualAnswerGeneratorState,
    _grade_generation_v_docuemnts_and_question_answer,
    _grade_generation_v_documents_and_question_hallucination,
    generate,
    grade_context,
    grade_generation,
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
@patch("retrieval_agents.modules.contextual_answer_generator.ChatPromptTemplate")
@patch("retrieval_agents.modules.contextual_answer_generator.load_chat_model")
@mark.parametrize(
    "input_documents, expected_documents, expected_goto, expected_finish_reason",
    [
        (
            [Document(page_content="relevant"), Document(page_content="relevant")],
            [Document(page_content="relevant"), Document(page_content="relevant")],
            "generate",
            None,
        ),
        (
            [Document(page_content="relevant"), Document(page_content="irrelevant")],
            [Document(page_content="relevant")],
            "generate",
            None,
        ),
        (
            [Document(page_content="irrelevant"), Document(page_content="irrelevant")],
            None,
            "__end__",
            "no_relevant_documents",
        ),
    ],
)
async def test_grade_context(
    mock_load_chat_model: MagicMock,
    mock_prompt_cls: MagicMock,
    runnable_config: RunnableConfig,
    input_documents: list[Document],
    expected_documents: list[Document],
    expected_goto: str,
    expected_finish_reason: str,
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

    state = ContextualAnswerGeneratorState(
        question="agent memory",
        # documents=[
        #    Document(page_content="relevant"),
        #    Document(page_content="irrelevant"),
        # ],
        documents=input_documents,
    )
    actual = await grade_context(state=state, config=runnable_config)
    assert actual.goto == expected_goto
    assert cast(dict[str, str], actual.update).get("question") == state.question
    assert (
        cast(dict[str, Sequence[Document]], actual.update).get("documents")
        == expected_documents
    )
    assert (
        cast(dict[str, str], actual.update).get("finish_reason")
        == expected_finish_reason
    )


@mark.asyncio
@patch("retrieval_agents.modules.contextual_answer_generator.ChatPromptTemplate")
@patch("retrieval_agents.modules.contextual_answer_generator.load_chat_model")
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
        state=ContextualAnswerGeneratorState(question="agent memory", documents=[]),
        config=runnable_config,
    )
    assert response["documents"] == []
    assert response["question"] == "agent memory"
    assert response["generation"] == "generated text"


### Edges ###


@mark.asyncio
@patch(
    "retrieval_agents.modules.contextual_answer_generator._grade_generation_v_documents_and_question_hallucination",
    new_callable=AsyncMock,
)
@patch(
    "retrieval_agents.modules.contextual_answer_generator._grade_generation_v_docuemnts_and_question_answer",
    new_callable=AsyncMock,
)
@mark.parametrize(
    "hallucination_grade, answer_grade, expected_goto, expected_finish_reason",
    [
        (True, True, "__end__", "complete"),
        (True, False, "__end__", "not_useful"),
        (False, True, "generate", None),
    ],
)
async def test_grade_generation(
    mock_grade_generation_v_docuemnts_and_question_answer: MagicMock,
    mock_grade_generation_v_documents_and_question_hallucination: MagicMock,
    runnable_config: RunnableConfig,
    hallucination_grade: bool,
    answer_grade: bool,
    expected_goto: str,
    expected_finish_reason: str | None,
) -> None:
    runnable_config["configurable"]["max_generation"] = 3
    mock_grade_generation_v_docuemnts_and_question_answer.return_value = answer_grade
    mock_grade_generation_v_documents_and_question_hallucination.return_value = (
        hallucination_grade
    )
    actual = await grade_generation(
        state=ContextualAnswerGeneratorState(question="", documents=[]),
        config=runnable_config,
    )
    assert actual.goto == expected_goto
    assert (
        cast(dict[str, str], actual.update).get("finish_reason")
        == expected_finish_reason
    )


@mark.asyncio
@patch("retrieval_agents.modules.contextual_answer_generator.ChatPromptTemplate")
@patch(
    "retrieval_agents.modules.contextual_answer_generator.load_chat_model",
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
    actual = await _grade_generation_v_documents_and_question_hallucination(
        state=ContextualAnswerGeneratorState(question="", documents=[]),
        configuration=MagicMock(),
    )
    assert actual == expected


@mark.asyncio
@patch("retrieval_agents.modules.contextual_answer_generator.ChatPromptTemplate")
@patch(
    "retrieval_agents.modules.contextual_answer_generator.load_chat_model",
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
    actual = await _grade_generation_v_docuemnts_and_question_answer(
        state=ContextualAnswerGeneratorState(question="", documents=[]),
        configuration=MagicMock(),
    )
    assert actual == expected
