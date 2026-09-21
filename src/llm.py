from functools import lru_cache
from langchain_ollama import ChatOllama
from .config import settings
@lru_cache(maxsize=1)
def get_llm(): return ChatOllama(model=settings.ollama_model,temperature=0)
