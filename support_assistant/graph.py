from pathlib import Path
import os

import chromadb
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from models import AnswerResponse

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[str]
    sources: list[str]
    answer: str
    confidence: float


model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    COLLECTION_NAME
)


SYSTEM_PROMPT = """
Role: You are a Zepto customer support assistant.

Context: Use the retrieved Zepto policy context to answer
policy-related questions.

Task: Answer the customer's question using only the
retrieved policy context.

Format: Give a concise, direct answer and identify the
source documents.

Length: Keep the response short and relevant.

Negative constraint: Do not invent policies, fees, timelines,
refunds, or other information that is not present in the
retrieved context.

Few-shot example:

Customer:
How long do I have to report a damaged item?

Context:
Customers must report damaged items within 24 hours of delivery.

Answer:
Based on the retrieved context: Customers must report
damaged items within 24 hours of delivery.
"""


def classify_intent(state: AssistantState):
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "track",
        "cancel",
        "cancellation",
        "gift card",
        "support",
        "damaged",
        "damage",
        "missing",
        "spoiled",
        "item",
        "order",
        "packed",
        "rider",
        "fee",
        "priority",
    ]

    is_policy = any(
        keyword in query
        for keyword in policy_keywords
    )

    return {
        "intent": "policy" if is_policy else "general"
    }


def retrieve_and_answer(state: AssistantState):
    query = state["query"]

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    top_chunk = documents[0]

    mock_llm = os.getenv("MOCK_LLM", "1") == "1"

    if mock_llm:
        answer = f"Based on the retrieved context: {top_chunk}"
    else:
        answer = f"Based on the retrieved context: {top_chunk}"
    sources = list(
        dict.fromkeys(
            metadata["source"]
            for metadata in metadatas
        )
    )

    response = AnswerResponse(
        answer=answer,
        sources=sources,
        confidence=0.9,
    )

    return {
        "retrieved_chunks": documents,
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
    }


def direct_answer(state: AssistantState):
    answer = (
        "I can help with Zepto support questions. "
        "Please ask about delivery, returns, refunds, "
        "membership, tracking, cancellation, gift cards, "
        "or customer support."
    )

    response = AnswerResponse(
        answer=answer,
        sources=[],
        confidence=0.8,
    )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }


def route_intent(state: AssistantState):
    if state["intent"] == "policy":
        return "retrieve_and_answer"

    return "direct_answer"


builder = StateGraph(AssistantState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()