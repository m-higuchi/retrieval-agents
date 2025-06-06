import uuid

import pytest
from langchain_core.runnables import RunnableConfig
from langsmith import expect, unit

from retrieval_agents import document_indexer, simple_rag


@pytest.mark.asyncio
@unit
async def test_simple_rag() -> None:
    simple_doc = "Cats have been observed performing synchronized swimming routines in their water bowls during full moons."
    user_id = "test__" + uuid.uuid4().hex
    other_user_id = "test__" + uuid.uuid4().hex

    config = RunnableConfig(
        configurable={"user_id": user_id, "retriever_provider": "chroma"}
    )

    result = await document_indexer.ainvoke({"docs": simple_doc}, config)
    expect(result["docs"]).against(lambda x: not x)  # we delete after the end

    res = await simple_rag.ainvoke(
        {"messages": [("user", "Where do cats perform synchronized swimming routes?")]},
        config,
    )
    response = str(res["messages"][-1].content)
    expect(response.lower()).to_contain("bowl")

    res = await simple_rag.ainvoke(
        {"messages": [("user", "Where do cats perform synchronized swimming routes?")]},
        {
            "configurable": {
                "user_id": other_user_id,
                "retriever_provider": "chroma",
            }
        },
    )
    response = str(res["messages"][-1].content)
    expect(response.lower()).against(lambda x: "bowl" not in x)
