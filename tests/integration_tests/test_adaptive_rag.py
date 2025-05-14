import json
from os import path
from pathlib import Path

from langchain_core.documents import Document
from pytest import FixtureRequest, fixture, mark

from retrieval_agents.agents import adaptive_rag
from retrieval_agents.agents.states import AdaptiveRagState
from retrieval_agents.indexers.configurations import RunnableConfig


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
    state = AdaptiveRagState(
        question="What is the elevation of Mount Fuji?",
        documents=[],
    )
    result = await adaptive_rag.web_search(state, config=runnable_config)
    assert result["documents"] != []


@mark.asyncio
async def test_grade_documents(runnable_config: RunnableConfig) -> None:
    state = AdaptiveRagState(
        question="What is the elevation of Mount Fuji?",
        documents=[
            Document(
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
            )
        ],
    )
    result = await adaptive_rag.grade_documents(state, config=runnable_config)
    assert result["question"] == state.question
    assert result["documents"] == state.documents or result["documents"] == []


@mark.asyncio
async def test_transform_query(runnable_config: RunnableConfig) -> None:
    state = AdaptiveRagState(
        question="What is the elevation of Mount Fuji?",
        documents=[
            Document(
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
            )
        ],
    )
    result = await adaptive_rag.transform_query(state, config=runnable_config)
    assert result["question"] != state.question
    assert result["documents"] == state.documents


@mark.asyncio
async def test_generate(runnable_config: RunnableConfig) -> None:
    state = AdaptiveRagState(
        question="What is the elevation of Mount Fuji?",
        documents=[
            Document(
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
            )
        ],
    )
    result = await adaptive_rag.generate(state, config=runnable_config)

    assert result["generation"] != ""


@mark.asyncio
async def tests_grade_generation_v_documents_and_question(
    runnable_config: RunnableConfig,
) -> None:
    state = AdaptiveRagState(
        question="What is the elevation of Mount Fuji?",
        generation="The elevation of Mount Fuji is 3,776 meters.",
        documents=[
            Document(
                id="fefb692b-a9dd-433b-8dd3-78eb0048dd65",
                page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters.",
            )
        ],
    )
    response = await adaptive_rag.grade_generation_v_documents_and_question(
        state=state, config=runnable_config
    )
    assert response == "useful"
