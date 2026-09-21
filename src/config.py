from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "meal-plan-rag")
    cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    region: str = os.getenv("PINECONE_REGION", "us-east-1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.20"))
settings = Settings()
