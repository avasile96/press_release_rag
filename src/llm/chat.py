from langchain_community.chat_models import ChatOllama
from src.config.settings import settings

"""
Chat wrapper (temperature, tokens).
"""

def get_chat():
    return ChatOllama(
        model=settings.ollama_chat_model,
        temperature=settings.temperature,
        num_predict=settings.max_tokens,
    )
