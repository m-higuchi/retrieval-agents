import json
from os import path
from pathlib import Path
from typing import Sequence, cast

from langchain_core.documents import Document
from pytest import FixtureRequest, fixture, mark

from retrieval_agents import RunnableConfig
from retrieval_agents.modules.contextual_answer_generator import (
    ContextualAnswerGeneratorState,
    generate,
    grade_context,
    grade_generation,
)


@fixture(params=["ollama", "openai", "anthropic"])
def runnable_config(request: FixtureRequest) -> RunnableConfig:
    with open(
        path.join(
            Path(__file__).parent.parent,
            "configurations/adaptive_rag",
            f"{request.param}.config",
        ),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    return RunnableConfig(configurable=data)


@mark.asyncio
async def test_grade_context(runnable_config: RunnableConfig) -> None:
    state = ContextualAnswerGeneratorState(
        question="What is the elevation of Mount Fuji?",
        documents=[
            Document(
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
            )
        ],
    )
    expected_values: list[dict[str, None | str | Sequence[Document]]] = [
        {
            "goto": "__end__",
            "finish_reason": "no_relevant_documents",
            "documents": None,
        },
        {"goto": "generate", "finish_reason": None, "documents": state.documents},
    ]
    actual = await grade_context(state, config=runnable_config)
    assert any(
        actual.goto == e["goto"]
        and cast(dict[str, str], actual.update).get("finish_reason")
        == e["finish_reason"]
        and cast(dict[str, Sequence[Document]], actual.update).get("documents")
        == e["documents"]
        for e in expected_values
    )


@mark.asyncio
async def test_generate(runnable_config: RunnableConfig) -> None:
    state = ContextualAnswerGeneratorState(
        question="What is the elevation of Mount Fuji?",
        documents=[
            Document(
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
            )
        ],
    )
    actual = await generate(state, config=runnable_config)

    assert (
        actual["question"] == state.question
        and actual["documents"] == state.documents
        and actual["generation"] != ""
    )


@mark.asyncio
async def tests_grade_generation(
    runnable_config: RunnableConfig,
) -> None:
    state = ContextualAnswerGeneratorState(
        question="What is the elevation of Mount Fuji?",
        generation="The elevation of Mount Fuji is 3,776 meters.",
        documents=[
            Document(
                id="fefb692b-a9dd-433b-8dd3-78eb0048dd65",
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters.",
            )
        ],
    )
    actual = await grade_generation(state=state, config=runnable_config)
    expected_values: list[dict[str, None | str | Sequence[Document]]] = [
        {
            "goto": "generate",
            "question": state.question,
            "documents": state.documents,
            "finish_reason": None,
        },
        {
            "goto": "__end__",
            "question": state.question,
            "documents": state.documents,
            "finish_reason": "not_useful",
        },
        {
            "goto": "__end__",
            "question": state.question,
            "documents": state.documents,
            "finish_reason": "complete",
        },
    ]
    assert any(
        actual.goto == e["goto"]
        and cast(dict[str, str], actual.update)["question"] == e["question"]
        and cast(dict[str, Sequence[Document]], actual.update)["documents"]
        == e["documents"]
        and cast(dict[str, str], actual.update).get("finish_reason")
        == e["finish_reason"]
        for e in expected_values
    )
