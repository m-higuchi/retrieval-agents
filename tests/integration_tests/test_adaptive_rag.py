import pytest
from langchain_core.documents import Document
from pytest import fixture, mark

from retrieval_agents.agents import adaptive_rag
from retrieval_agents.agents.configuration import RunnableConfig
from retrieval_agents.agents.state import ARagState


@fixture
def runnable_config():
    return RunnableConfig(
        user_id="test_user",
        hallucination_grader_model="ollama/llama3.2:3b-instruct-fp16",
        answer_grader_model="ollama/llama3.2:3b-instruct-fp16",
    )


async def test_generate(runnable_config):
    # response = await adaptive_rag.generate(runnable_config)

    # question = "What is the elevation of Mount Fuji?"
    # generation = "The elevation of Mount Fuji is 3,776 meters."
    # docs = [
    #    Document(
    #        page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters."
    #    )
    # ]
    pytest.fail()


@mark.asyncio
async def tests_grade_generation_v_documents_and_question(
    runnable_config: RunnableConfig,
):
    generation = "The elevation of Mount Fuji is 3,776 meters."
    documents = [
        Document(
            id="fefb692b-a9dd-433b-8dd3-78eb0048dd65",
            page_content="Mount Fuji is the tallest mountain in Japan, with an elevation of 3,776 meters.",
        )
    ]
    state = ARagState(
        question="What is the elevation of Mount Fuji?",
        generation=generation,
        documents=documents,
    )
    response = await adaptive_rag.grade_generation_v_documents_and_question(
        state=state, config=runnable_config
    )
    assert response == "useful"
