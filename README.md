# Press Release RAG Project

A minimal Retrieval-Augmented Generation (RAG) system for press releases.

## Project Structure

```
press_release_rag/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_ingestion.py   # Document loading and preprocessing
│   ├── embedding.py        # Document embedding and vector operations
│   ├── retrieval.py        # Document retrieval and similarity search
│   ├── generation.py       # Text generation with retrieved context
│   └── rag_pipeline.py     # Main RAG pipeline orchestration
├── data/                   # Data storage
│   ├── raw/               # Raw input documents
│   └── processed/         # Processed and indexed documents
├── config/                # Configuration files
│   └── config.py          # System configuration settings
└── tests/                 # Test suite
    ├── __init__.py
    ├── test_data_ingestion.py
    └── test_rag_pipeline.py
```

## Setup

TODO: Add setup instructions

## Usage

TODO: Add usage instructions