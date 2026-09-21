import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import streamlit as st
from src.rag_graph import ask
from src.config import settings

st.set_page_config(page_title="Meal Plan RAG Assistant",page_icon="🥗",layout="wide")
st.title("🥗 Meal Plan RAG Assistant")
st.caption("Recipe + nutrition RAG: LangChain/LangGraph • Pinecone • local embeddings • cited answers")
with st.sidebar:
    st.subheader("RAG settings")
    st.write(f"Top-K: **{settings.top_k}**")
    st.write(f"Embedding: `{settings.embedding_model}`")
    st.write(f"Generator: `{settings.ollama_model}`")
    st.info("Answers are grounded in retrieved recipe records. Dietary/allergen safety is not guaranteed.")
    if st.button("Clear chat"): st.session_state.messages=[]

if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if q:=st.chat_input("Ask about recipes, diet labels, ingredients, calories, or protein…"):
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.status("Retrieving recipes and generating a grounded answer…"):
                result=ask(q)
            st.markdown(result["answer"])
            if result.get("sources"):
                st.markdown("**Retrieved sources**")
                for s in result["sources"]:
                    label=f"[{s['marker']}] {s['recipe_name']} — {s['source']} (score {s['score']})"
                    if s.get("url"): st.markdown(f"- [{label}]({s['url']})")
                    else: st.markdown(f"- {label}")
            answer=result["answer"]
        except Exception as e:
            answer=f"Configuration/runtime error: `{e}`"
            st.error(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
