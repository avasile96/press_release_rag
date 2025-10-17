# Press Release RAG Project

A minimal Retrieval-Augmented Generation (RAG) system for press releases.

## Project Structure

press_release_rag
├─ pyproject.toml
├─ README.md
├─ .env.example
├─ Makefile
├─ docker-compose.yml
│
├─ data/
│  ├─ raw/                  # your original .txt / .jsonl / .csv
│  ├─ interim/              # chunked/normalized text (optional)
│  └─ vectorstore/          # FAISS index files (*.faiss, *.pkl)
│
├─ src/
│  ├─ config/
│  │  └─ settings.py        # Pydantic settings (models, paths, params)
│  ├─ models/
│  │  └─ schemas.py         # Pydantic models for docs, chunks, query, answer
│  ├─ ingest/
│  │  ├─ loaders.py         # read .txt/.jsonl, split into docs
│  │  ├─ chunkers.py        # LangChain text splitters (chars/tokens)
│  │  └─ build_index.py     # create/update FAISS index with Ollama embeddings
│  ├─ retrieval/
│  │  ├─ retriever.py       # FAISS retriever factory (k, filters)
│  │  └─ prompt.py          # system/user templates for RAG
│  ├─ llm/
│  │  ├─ chat.py            # Chat model (Ollama), temperature, tools
│  │  └─ embeddings.py      # Embedding model (Ollama)
│  ├─ rag/
│  │  └─ chain.py           # build RAG chain (retrieve → condense → answer)
│  ├─ eval/
│  │  ├─ goldsets.py        # small hand-made Q/A for smoke tests
│  │  └─ evaluate.py        # latency@k, hit@k, context precision/len
│  └─ utils/
│     └─ io.py              # small helpers (save/load, path utils, timing)
│
├─ app/
│  └─ streamlit_app.py      # simple UI: ask question, see sources & answer
│
└─ scripts/
   ├─ ingest.py             # CLI: build/refresh index
   ├─ query.py              # CLI: one-off question from terminal
   └─ dump_sources.py       # debug: inspect top-k 