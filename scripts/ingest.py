from pathlib import Path
from src.config.settings import settings
from src.ingest.loaders import load_plaintext
from src.ingest.chunkers import chunk_documents
from src.ingest.build_index import build_faiss

def main():
    # Put your simulated file here, or pass path via env/argparse
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
