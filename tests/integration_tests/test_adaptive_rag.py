import json
from os import path
from pathlib import Path

from langchain_core.documents import Document
from pytest import FixtureRequest, fixture, mark

from retrieval_agents import RunnableConfig
from retrieval_agents.modules.adaptive_rag import (
    ContextualAnswerGeneratorState,
    transform_query,
    web_search,
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
async def test_web_search(runnable_config: RunnableConfig) -> None:
    state = ContextualAnswerGeneratorState(
        question="What is the elevation of Mount Fuji?",
        documents=[],
    )
    actual = await web_search(state, config=runnable_config)
    assert actual["question"] == state.question and actual["documents"] != []


@mark.asyncio
async def test_transform_query(runnable_config: RunnableConfig) -> None:
    state = ContextualAnswerGeneratorState(
        question="What is the elevation of Mount Fuji?",
        documents=[
            Document(
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
            )
        ],
    )
    result = await transform_query(state, config=runnable_config)
    assert (
        result["question"] != state.question and result["documents"] == state.documents
    )
