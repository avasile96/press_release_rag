from langchain_community.chat_models import ChatOllama
from src.config.settings import settings

"""
Chat wrapper (temperature, tokens).
"""

def get_chat():
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_host,  # <- important
        temperature=settings.temperature,
        num_predict=settings.max_tokens,
    )
