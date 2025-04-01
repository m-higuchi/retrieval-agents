import os
import sys
from dataclasses import asdict

from langsmith import aevaluate

from retrieval_agents.agents.adaptive_rag import graph

sys.path.append(os.path.dirname(__file__))
import asyncio

from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from ls_datasets.bootcamp_qa import create_dataset, get_source_documents  # type: ignore

from retrieval_agents.agents.configuration import ARagConfiguration

configuration = ARagConfiguration(
    user_id="test_user",
    embedding_model="nomic/nomic-embed-text-v1.5",
    answer_grader_model="anthropic/claude-3-5-haiku-20241022",
    router_model="anthropic/claude-3-5-haiku-20241022",
    rewrite_model="anthropic/claude-3-7-sonnet-20250219",
    generate_model="anthropic/claude-3-7-sonnet-20250219",
    grade_documents_model="anthropic/claude-3-5-haiku-20241022",
    hallucination_grader_model="anthropic/claude-3-5-haiku-20241022",
    retriever_provider="chroma",
    topics="all topics",
)


async def evaluatie(evaluators):
    dataset = create_dataset()
    docs = get_source_documents({"user_id": configuration.user_id})
    vstore = Chroma.from_documents(
        documents=docs,
        embedding=NomicEmbeddings(
            model=str.split(configuration.embedding_model, "/")[1]
        ),
        collection_name=os.environ["CHROMA_COLLECTION_NAME"],
        persist_directory=os.environ["CHROMA_DIR"],
    )

    result = await aevaluate(
        ainvoke_wrapper, data=dataset.name, evaluators=evaluators, max_concurrency=1
    )


async def ainvoke_wrapper(example: dict):
    input_state = ARagInputState(question=example["question"])
    state_dict = await graph.ainvoke(input_state, config=asdict(configuration))
    return {
        "answer": state_dict["generation"],
        "contexts": state_dict["documents"],
        "user_input": state_dict["question"],
        "response": state_dict["generation"],
        "retrieved_contexts": state_dict["documents"],
    }


if __name__ == "__main__":
    import os

    from ragas.integrations.langchain import EvaluatorChain
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from retrieval_agents.agents.state import ARagInputState

    os.environ["CHROMA_DIR"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./.data"
    )
    os.environ["CHROMA_COLLECTION_NAME"] = "retrieval-agents"

    metrics = [
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ]
    evaluators = [EvaluatorChain(metric).evaluate_run for metric in metrics]
    result = asyncio.run(evaluatie(evaluators))
    print(result)
