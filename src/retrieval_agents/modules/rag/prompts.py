"""Default prompts."""

RESPONSE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrieved_docs}

System time: {system_time}"""
QUERY_SYSTEM_PROMPT = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""

GENERATE_HUMAN_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

ROUTER_SYSTEM_PROMPT = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to {topics}.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

REWRITE_SYSTEM_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

REWRITE_HUMAN_PROMPT = """Here is the initial question: {question}

Formulate an improved question."""

HALLUCINATION_GRADER_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

HALLUCINATION_GRADER_HUMAN_PROMPT = """Set of facts: {documents} 

LLM generation: {generation}"""

GRADE_DOCUMENTS_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

GRADE_DOCUMENTS_HUMAN_PROMPT = (
    """Retrieved document: \n\n {document} \n\n User question: {question}"""
)

ANSWER_GRADER_SYSTEM_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

ANSWER_GRADER_HUMAN_PROMPT = (
    """User question: \n\n {question} \n\n LLM generation: {generation}"""
)
