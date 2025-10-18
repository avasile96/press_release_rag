from langchain_ollama import OllamaEmbeddings
from src.config.settings import settings

def get_embeddings():
    """Return an embeddings object for generating vector embeddings.

    Returns
    -------
    Embeddings
        A LangChain-compatible embeddings instance that implements
        `.embed_documents` / `.embed_query` or similar. The returned object
        is configured using settings (model name and host URL).
    """

    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_host,  # <- important
    )
