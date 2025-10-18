from pathlib import Path
from src.config.settings import settings
from src.ingest.loaders import load_plaintext
from src.ingest.chunkers import chunk_documents
from src.ingest.build_index import build_faiss

def main():
    """Main CLI entrypoint for ingesting raw text files into FAISS.

    The function finds all `.txt` files in the configured `data_raw_dir`,
    converts them to Document objects, chunks them, builds a FAISS index,
    and persists the index to the configured `vectorstore_dir`.
    """

    raw_paths = list(settings.data_raw_dir.glob("*.txt"))
    assert raw_paths, f"No .txt files in {settings.data_raw_dir}"
    docs = []
    for p in raw_paths:
        docs.extend(load_plaintext(p))
    chunks = chunk_documents(docs)
    build_faiss(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs.")

if __name__ == "__main__":
    main()
