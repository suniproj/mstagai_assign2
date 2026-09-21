from pinecone import Pinecone
from .config import settings
pc=Pinecone(api_key=settings.pinecone_api_key)
if settings.index_name in [x.name for x in pc.list_indexes()]:
    pc.delete_index(settings.index_name); print("Deleted",settings.index_name)
else: print("No existing index")
