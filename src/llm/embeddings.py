from langchain_community.embeddings import OllamaEmbeddings
from src.config.settings import settings

def get_embeddings():
    return OllamaEmbeddings(model=settings.ollama_embed_model)
