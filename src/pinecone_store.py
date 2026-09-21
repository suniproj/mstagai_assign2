import time
from pinecone import Pinecone,ServerlessSpec
from .config import settings
from .embeddings import dimension
def index():
    if not settings.pinecone_api_key: raise RuntimeError("Set PINECONE_API_KEY in .env")
    pc=Pinecone(api_key=settings.pinecone_api_key)
    names=[x.name for x in pc.list_indexes()]
    if settings.index_name not in names:
        pc.create_index(name=settings.index_name,dimension=dimension(),metric="cosine",spec=ServerlessSpec(cloud=settings.cloud,region=settings.region))
        while not pc.describe_index(settings.index_name).status["ready"]: time.sleep(1)
    return pc.Index(settings.index_name)
