import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.retriever import retrieve
def main():
    path=ROOT/"data/eval/golden_questions.csv"; rows=list(csv.DictReader(path.open()))
    hits=0
    for r in rows:
        names=[d.metadata.get("recipe_name","") for d,_ in retrieve(r["question"])]
        ok=r["expected_recipe"].lower() in [x.lower() for x in names]; hits+=ok
        print(("PASS" if ok else "FAIL"),r["question"],"->",names)
    print(f"Recall@K: {hits}/{len(rows)} = {hits/len(rows):.1%}" if rows else "No eval rows.")
if __name__=="__main__": main()
