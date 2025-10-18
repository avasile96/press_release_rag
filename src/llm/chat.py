from langchain_community.chat_models import ChatOllama
from src.config.settings import settings

"""
Chat wrapper (temperature, tokens).
"""

def get_chat():
    """Return a configured chat model instance for text generation.

    Returns
    -------
    ChatModel
        A chat model instance compatible with LangChain runnables. It is
        configured using values from `settings` such as model name,
        base_url, temperature and token limits.
    """

    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_host,
        temperature=settings.temperature,
        num_predict=settings.max_tokens,
    )
