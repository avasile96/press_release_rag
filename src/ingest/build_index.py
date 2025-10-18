from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.llm.embeddings import get_embeddings
from src.config.settings import settings

"""
Create the FAISS store from chunks
"""

def build_faiss(chunks: list) -> None:
    emb = get_embeddings()

    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, **c.meta} for c in chunks]

    # normalize_L2=True → cosine similarity via inner product on unit vectors
    vs = FAISS.from_texts(texts, emb, metadatas=metadatas, normalize_L2=True)

    Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(str(settings.vectorstore_dir))
