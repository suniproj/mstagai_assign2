import argparse,pandas as pd
from .documents import make_documents
from .embeddings import get_embeddings
from .pinecone_store import index
def ingest(path,limit=None,batch_size=64):
    docs=make_documents(pd.read_csv(path),limit); emb=get_embeddings(); ix=index()
    for start in range(0,len(docs),batch_size):
        batch=docs[start:start+batch_size]; vecs=emb.embed_documents([d.page_content for d in batch]); payload=[]
        for d,v in zip(batch,vecs):
            md=dict(d.metadata); md["text"]=d.page_content
            payload.append({"id":f"recipe-{md['row_id']}","values":v,"metadata":md})
        ix.upsert(vectors=payload); print(f"Upserted {min(start+batch_size,len(docs))}/{len(docs)}")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv",required=True); p.add_argument("--limit",type=int); a=p.parse_args(); ingest(a.csv,a.limit)
if __name__=="__main__": main()
