from fastapi import FastAPI
from pydantic import BaseModel

from graph import graph
app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto customer support assistant",
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant API is running"
    }


@app.post("/ask")
def ask(request: AskRequest):
    result = graph.invoke({
        "query": request.query
    })

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
    }