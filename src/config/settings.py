# Configuration settings for the RAG system
from pydantic_settings import BaseSettings
from pathlib import Path

"""
Centralization of everything that changes between laptops/envs 
(model names, chunk sizes, FAISS paths, k, temperature, etc.)
"""

class Settings(BaseSettings):
    # Files & paths
    data_raw_dir: Path = Path("data/raw")
    vectorstore_dir: Path = Path("data/vectorstore")

    # Models (override in .env)
    ollama_host: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4

    # Generation
    temperature: float = 0.2
    max_tokens: int = 1024

    class ConfigDict:
        env_file = ".env"

settings = Settings()
