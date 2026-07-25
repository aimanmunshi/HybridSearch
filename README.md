# Semantic Search Engine

Hybrid (vector + keyword) semantic search over a recipe dataset, built with FastAPI, pgvector, and React.

> Work in progress — full documentation (architecture, setup, eval results, tradeoffs) lands in Phase 10.

## Stack

- **Backend:** FastAPI
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`), OpenAI `text-embedding-3-small` optional
- **Vector store:** Postgres + pgvector
- **Keyword search:** Postgres full-text search (`tsvector`)
- **Frontend:** React (Vite) + Tailwind CSS
- **Containerization:** Docker Compose

## Quickstart (local dev, once later phases land)

```bash
cp .env.example .env
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173
