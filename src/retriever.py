from langchain_core.documents import Document
from .config import settings
from .embeddings import get_embeddings
from .pinecone_store import index
def retrieve(query,k=None):
    result=index().query(vector=get_embeddings().embed_query(query),top_k=k or settings.top_k,include_metadata=True)
    out=[]
    for m in result.matches:
        md=dict(m.metadata or {}); text=md.pop("text","")
        out.append((Document(page_content=text,metadata=md),float(m.score)))
    return out
