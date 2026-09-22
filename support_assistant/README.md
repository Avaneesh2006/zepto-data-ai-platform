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