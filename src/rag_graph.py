from pathlib import Path
from typing import TypedDict,Any
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,HumanMessage
from .config import settings
from .retriever import retrieve
from .llm import get_llm
PROMPT=Path(__file__).resolve().parents[1]/"prompts"/"rag_system_prompt_v2.md"
class State(TypedDict,total=False):
    question:str; retrieved:list[Any]; sufficient:bool; answer:str; sources:list[dict]
def rnode(s): return {"retrieved":retrieve(s["question"])}
def gate(s):
    r=s.get("retrieved",[]); return {"sufficient":bool(r) and max(score for _,score in r)>=settings.min_relevance_score}
def route(s): return "generate" if s.get("sufficient") else "refuse"
def refuse(s): return {"answer":"I couldn't find enough relevant evidence in the recipe corpus to answer that reliably.","sources":[]}
def generate(s):
    blocks=[]; sources=[]
    for i,(doc,score) in enumerate(s["retrieved"],1):
        mark=f"R{i}"; blocks.append(f"[{mark}] similarity={score:.3f}\n{doc.page_content}")
        sources.append({"marker":mark,"recipe_name":doc.metadata.get("recipe_name","Unknown"),"source":doc.metadata.get("source","Unknown"),"url":doc.metadata.get("url",""),"score":round(score,3)})
    system=PROMPT.read_text().replace("{context}","\n\n".join(blocks))
    msg=get_llm().invoke([SystemMessage(content=system),HumanMessage(content=s["question"])])
    return {"answer":msg.content,"sources":sources}
g=StateGraph(State); g.add_node("retrieve",rnode); g.add_node("gate",gate); g.add_node("generate",generate); g.add_node("refuse",refuse)
g.add_edge(START,"retrieve"); g.add_edge("retrieve","gate"); g.add_conditional_edges("gate",route,{"generate":"generate","refuse":"refuse"}); g.add_edge("generate",END); g.add_edge("refuse",END)
GRAPH=g.compile()
def ask(q): return GRAPH.invoke({"question":q})
