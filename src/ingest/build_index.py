import faiss
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from src.llm.embeddings import get_embeddings
from src.config.settings import settings

"""
Create the FAISS store from chunks
"""

def build_faiss(chunks: list) -> None:
    """Build and persist a FAISS vector store from chunks.

    Parameters
    ----------
    chunks : list
        An iterable of chunk-like objects with attributes `text`, `doc_id`,
        and `meta` (metadata mapping). The function will compute embeddings
        for each chunk, create a FAISS vector store and save it to the
        configured `vectorstore_dir`.

    Notes
    -----
    This function uses `get_embeddings()` to obtain an embeddings object
    compatible with the LangChain FAISS wrapper and saves the index locally.
    """

    emb = get_embeddings()

    # index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))

    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, **c.meta} for c in chunks]

    # Defaults to Euclidean distance
    vs = FAISS.from_texts(texts, 
                          emb,
                        #   faiss.IndexFlatL2,
                        #   emb, 
                          metadatas=metadatas, 
                        #   normalize_L2=True,
                          distance_strategy=DistanceStrategy.COSINE,
                          )

    
    # print(vs.index)

    Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(str(settings.vectorstore_dir))
