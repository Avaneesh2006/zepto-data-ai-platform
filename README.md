# Zepto Data & AI Platform

An end-to-end AI/ML engineering capstone project covering data engineering, analytics, machine learning, RAG, LangGraph, and API deployment.

## Project Overview

This project is divided into three modules:

1. **Data Pipeline** — Web scraping, data cleaning, SQLite database design, SQL analytics, and Pandas validation.
2. **Analytics & Machine Learning** — Titanic EDA, visualization, classification, imbalance handling, hyperparameter tuning, and fare regression.
3. **GenAI Support Assistant** — Policy-document RAG using embeddings, ChromaDB, LangGraph, Pydantic structured outputs, FastAPI, and Docker.

## Project Structure

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── pipeline.py
│   ├── database.py
│   ├── queries.py
│   ├── raw_books.csv
│   ├── clean_books.csv
│   ├── books.db
│   ├── query_results.txt
│   └── README.md
│
├── analytics/
│   ├── eda.py
│   ├── modelling.py
│   ├── titanic.csv
│   ├── model_pipeline.joblib
│   ├── plots/
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   ├── ingestion.py
│   ├── graph.py
│   ├── models.py
│   ├── api.py
│   ├── test_retrieval.py
│   ├── test_graph.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── .gitignore
└── README.md