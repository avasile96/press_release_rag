# Press Release RAG

A minimal Retrieval-Augmented Generation (RAG) system for querying press releases using LangChain, FAISS, Ollama, and Streamlit. The repository demonstrates ingestion, semantic retrieval, and grounded answering with cited context, running either natively or with Docker Compose.

---

## Features

- Local LLM inference via Ollama (chat and embeddings)
- FAISS vector store with cosine similarity
- Modular pipeline: ingest → retrieve → generate
- Evaluation pipeline: BLEU, ROUGE-L + Faithfulness, Relevancy
- Streamlit interface that displays answers and retrieved sources
- Native run or Docker Compose; models and indexes persist via volumes

---

## Architecture

1. **Ingestion**: plaintext press releases are split into documents/chunks and embedded with Ollama embeddings.  
2. **Indexing**: embeddings are stored in a FAISS index on disk.  
3. **Retrieval**: a LangChain retriever fetches top-k chunks for a query.  
4. **Generation**: an Ollama chat model answers strictly from retrieved context; citations are shown in the UI.

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
```

---

## Configuration

Copy environment variables and adjust values:

```bash
cp .env.example .env
```

`.env.example`:

```ini
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=6
TEMPERATURE=0.2
MAX_TOKENS=1024
```

In Docker, `OLLAMA_HOST` is provided as `http://ollama:11434` via `docker-compose.yml`.

---

## Local Usage (no Docker)

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install pydantic streamlit faiss-cpu langchain>=0.2 langchain-community>=0.2 langchain-text-splitters>=0.2 langchain-ollama
```

Start Ollama and pull models:

```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

Place a plaintext corpus at `data/raw/press_releases.txt` (or any `*.txt`).

Build the FAISS index and start the UI:

```bash
python -m scripts.ingest
PYTHONPATH=$(pwd) python -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

---

## Docker Compose Usage

Start the stack (Ollama + app):

```bash
docker compose up --build -d
```

Pull models inside the Ollama container (first run only):

```bash
docker compose exec ollama ollama pull llama3
docker compose exec ollama ollama pull nomic-embed-text
```

The app ingests on start; view logs:

```bash
docker compose logs -f app
```

Open `http://localhost:8501`.

Stop services but keep models and index:

```bash
docker compose down
```

Remove containers and volumes (deletes models and index):

```bash
docker compose down -v
```

---

## Notable Implementation Details

- FAISS index is built with `normalize_L2=True`, making inner product equivalent to cosine similarity.
- For LangChain ≥ 0.2, retrievers implement the `Runnable` interface; the UI calls `retriever.invoke(question)`.
- On Apple Silicon, FAISS-CPU is used; FAISS GPU is not available with Metal.
- Docker Compose provides a private network so the app reaches Ollama at `http://ollama:11434`. Models persist in the named volume `ollama:/root/.ollama`.

---

## Troubleshooting

**Connection refused to 11434**
```bash
# Local: ensure Ollama is running
ollama serve

# Docker: confirm the app targets the service name
echo $OLLAMA_HOST  # should be http://ollama:11434 inside the app container
```

**ModuleNotFoundError: langchain.docstore.document**
```python
# Use FAISS.from_texts(...) instead of importing Document from langchain.docstore
# or import Document from:
from langchain_core.documents import Document
```

**ModuleNotFoundError: src when running scripts**
```bash
# Run from repository root
python -m scripts.ingest
# For Streamlit
PYTHONPATH=$(pwd) python -m streamlit run app/streamlit_app.py
```

**High Docker memory when idle**
```yaml
# docker-compose.yml (ollama service)
environment:
  OLLAMA_KEEP_ALIVE: "30s"
  OLLAMA_MAX_LOADED_MODELS: "1"
```

---

## License

MIT 
