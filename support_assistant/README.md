# Zepto Support Assistant

A RAG-based customer support assistant built using policy documents, sentence-transformer embeddings, ChromaDB, LangGraph, Pydantic, FastAPI, and Docker.

## Architecture

The system follows this flow:

```text
Policy Documents
      ↓
Text Chunking
      ↓
Sentence Transformer Embeddings
      ↓
ChromaDB Vector Store
      ↓
User Query
      ↓
Intent Classification
      ↓
Policy Retrieval / Direct Answer
      ↓
Pydantic Validation
      ↓
FastAPI Response


## Validation

The assistant was tested using both local FastAPI and Docker.

Docker deployment was verified with:

```powershell
docker build -t zepto-support-assistant ./support_assistant
docker run --rm -p 8000:8000 zepto-support-assistant