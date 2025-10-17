from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage
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

def system_message():
    return SystemMessage(content="You are a precise, concise enterprise assistant.")
