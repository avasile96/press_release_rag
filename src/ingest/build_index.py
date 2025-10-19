import faiss
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from src.llm.embeddings import get_embeddings
from src.config.settings import settings


def build_faiss(
    chunks: Iterable,
    *,
    index_factory: Optional[str] = None,
    metric: Optional[object] = None,
    normalize_L2: bool = False,
    mode: str = "default"
    ) -> None:
    """Build a FAISS index from chunks"""

    emb = get_embeddings()
    texts: List[str] = [c.text for c in chunks]
    metadatas: List[Mapping] = [{"doc_id": c.doc_id, **c.meta} for c in chunks]

    if not texts:
        raise ValueError("No chunks provided")

    embeddings = emb.embed_documents(texts)

    if mode == "default":
        vs = FAISS.from_texts(texts,
                          emb,
                          metadatas=metadatas,
                          distance_strategy=DistanceStrategy.COSINE)

        Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)
        vs.save_local(str(settings.vectorstore_dir))

    else:

        dim = len(embeddings[0])

        # Resolve metric (default: L2). Accept int (faiss constant) or simple strings.
        if metric is None:
            metric_const = faiss.METRIC_L2
        elif isinstance(metric, int):
            metric_const = metric
        elif isinstance(metric, str):
            s = metric.strip().lower()
            metric_const = faiss.METRIC_INNER_PRODUCT if s.startswith("ip") else faiss.METRIC_L2
        else:
            raise ValueError("metric must be None, int, or str")

        # Build index
        try:
            index = faiss.index_factory(dim, index_factory, metric_const) if index_factory else (
                faiss.IndexFlatIP(dim) if metric_const == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            )
        except Exception:
            index = faiss.IndexFlatL2(dim)

        vs = FAISS(emb, 
                    index, 
                    InMemoryDocstore(), 
                    {}, 
                    normalize_L2=normalize_L2)
        vs.add_embeddings(list(zip(texts, embeddings)), metadatas=metadatas)

        Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)
        vs.save_local(str(settings.vectorstore_dir))
