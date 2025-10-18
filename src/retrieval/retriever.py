from langchain_community.vectorstores import FAISS
from src.llm.embeddings import get_embeddings
from src.config.settings import settings

"""
Re-load FAISS and expose a LangChain retriever.
"""

def get_retriever():
    """Load a persisted FAISS vectorstore and return a LangChain retriever.

    Returns
    -------
    langchain.vectorstores.base.Retriever
        A retriever object configured with `search_kwargs` derived from
        settings (for example, `k` the number of neighbors to return).
    """

    vs = FAISS.load_local(
        str(settings.vectorstore_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": settings.top_k})

