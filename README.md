# Press Release RAG

A minimal Retrieval-Augmented Generation (RAG) system for querying press releases using LangChain, FAISS, Ollama, and Streamlit. The project demonstrates end-to-end ingestion, semantic retrieval, and grounded answering with cited context.

---

## Features

- Local LLM inference via Ollama (chat and embeddings).
- FAISS vector store with cosine similarity.
- Modular pipeline: ingest → retrieve → generate.
- Streamlit interface that displays answers and retrieved sources.
- Runs natively or with Docker Compose; models and indexes persist via volumes.

---

## Architecture

1. Ingestion: plaintext press releases are split into documents/chunks and embedded with Ollama embeddings.
2. Indexing: embeddings are stored in a FAISS index on disk.
3. Retrieval: a LangChain retriever fetches top-k chunks for a query.
4. Generation: an Ollama chat model answers strictly from retrieved context; citations are shown in the UI.

---

## Tech Stack

- Python 3.11+  
- LangChain (chains, retrievers)  
- FAISS-CPU (vector store)  
- Ollama (LLM/chat: `llama3`, embeddings: `nomic-embed-text`)  
- Streamlit (web UI)

---

## Repository Structure

```text
press_release_rag
├─ pyproject.toml
├─ README.md
├─ .env.example
├─ Makefile
├─ docker-compose.yml
│
├─ data/
│  ├─ raw/                  # original .txt/.jsonl/.csv
│  ├─ interim/              # optional normalized/chunked text
│  └─ vectorstore/          # FAISS index files (*.faiss, *.pkl)
│
├─ src/
│  ├─ config/
│  │  └─ settings.py        # Pydantic settings (paths, model names, params)
│  ├─ models/
│  │  └─ schemas.py         # Pydantic schemas for docs/chunks/query/answer
│  ├─ ingest/
│  │  ├─ loaders.py         # plaintext/jsonl readers
│  │  ├─ chunkers.py        # text splitting (token/char)
│  │  └─ build_index.py     # FAISS construction with embeddings
│  ├─ retrieval/
│  │  ├─ retriever.py       # FAISS retriever factory (k, filters)
│  │  └─ prompt.py          # system/user prompt templates
│  ├─ llm/
│  │  ├─ chat.py            # Chat model wrapper (Ollama)
│  │  └─ embeddings.py      # Embedding wrapper (Ollama)
│  ├─ rag/
│  │  └─ chain.py           # RAG chain (retrieve → format → generate)
│  ├─ eval/
│  │  ├─ goldsets.py        # small smoke-test Q/A sets
│  │  └─ evaluate.py        # latency@k, hit@k, context metrics
│  └─ utils/
│     └─ io.py              # helpers (paths, timing, I/O)
│
├─ app/
│  └─ streamlit_app.py      # UI: question input, answer, sources
│
└─ scripts/
   ├─ ingest.py             # build/refresh FAISS index
   ├─ query.py              # one-off terminal query
   └─ dump_sources.py       # inspect top-k retrieved chunks
