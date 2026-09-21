# Meal Plan RAG Assistant

RAG application for recipe, nutrition, and dietary questions using LangChain, LangGraph, Pinecone, local Sentence Transformer embeddings, Ollama/Llama 3.2 3B, and Streamlit.

## Code flow

### Ingestion
`recipes-with-nutrition.csv` -> `prepare_corpus.py` -> balanced 300-recipe corpus -> `documents.py` -> one LangChain Document per recipe -> `all-MiniLM-L6-v2` -> one 384-d vector per recipe -> `ingest.py` -> Pinecone.

### Query
User question -> same local embedding model -> Pinecone Top-K retrieval -> LangGraph evidence gate -> retrieved context + system prompt -> Ollama/Llama 3.2 3B -> cited answer -> Streamlit.

LangChain supplies Documents, embedding integration, and message primitives. LangGraph implements the small `retrieve -> evidence check -> generate/refuse` state flow.

## Project structure

```text
app/streamlit_app.py          Streamlit chat UI
data/raw/                     Source CSV (not committed)
data/processed/               Generated corpus (not committed)
data/eval/golden_questions.csv
evals/retrieval_eval.py       Recall@K evaluation
prompts/rag_system_prompt_v1.md
prompts/rag_system_prompt_v2.md
src/config.py
src/parsing.py
src/prepare_corpus.py         Builds 300-recipe corpus
src/documents.py              Creates LangChain Documents
src/embeddings.py             Local embeddings
src/pinecone_store.py         Pinecone index
src/ingest.py                 Embeds/upserts recipes
src/retriever.py              Dense retrieval + optional metadata filters
src/rag_graph.py              LangGraph workflow
src/llm.py                    Ollama integration
src/reset_index.py            Deletes index before rebuild
```

## Corpus

Source: 39,447 recipes. Quality checks retained 39,446. A reproducible sample (seed 42) creates:

| Group | Breakfast | Lunch/Dinner | Snack | Total |
|---|---:|---:|---:|---:|
| Vegan | 30 | 40 | 30 | 100 |
| Vegetarian-only | 30 | 40 | 30 | 100 |
| Non-Vegetarian | 30 | 40 | 30 | 100 |
| Total | 90 | 120 | 90 | 300 |

Non-vegetarian recipes provide negative examples for dietary retrieval tests.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set `PINECONE_API_KEY` in `.env`. Do not commit `.env`.

Install/pull the local generator:

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

## Prepare and ingest

Put the source CSV at `data/raw/recipes-with-nutrition.csv`.

```bash
.venv/bin/python -m src.prepare_corpus --input data/raw/recipes-with-nutrition.csv
.venv/bin/python -m src.reset_index
.venv/bin/python -m src.ingest --csv data/processed/recipes_clean.csv
```

Expected final ingestion output:

```text
Upserted 300/300
Ingestion complete: 300 recipes indexed.
```

## Run

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Example questions:
- Find me a high-protein vegan dinner.
- Find me a vegetarian breakfast.
- Find me a soy-free vegan meal.

## Tests

### Embedding smoke test
```bash
.venv/bin/python -c "from src.embeddings import get_embeddings; m=get_embeddings(); print(len(m.embed_query('vegan lentil dinner')))"
```
Expected: `384`.

### Dense retrieval baseline
```bash
.venv/bin/python -c 'from src.retriever import retrieve; r=retrieve("high protein vegan dinner"); print([(d.metadata["recipe_name"], round(s,3)) for d,s in r])'
```

Dense retrieval is intentionally retained as a baseline; it may return semantically related recipes that violate exact constraints.

### Metadata-filtered retrieval
For the experiment, "high protein" is defined as >=15g protein/serving.

```bash
.venv/bin/python -c 'from src.retriever import retrieve; f={"health_labels":{"$in":["Vegan"]},"meal_type":{"$in":["lunch/dinner"]},"protein_per_serving":{"$gte":15}}; r=retrieve("high protein vegan dinner", metadata_filter=f); print([(d.metadata["recipe_name"], d.metadata["protein_per_serving"], round(s,3)) for d,s in r])'
```

Observed Top-5:
- Vegetable Stir Fry — 15.52g
- Frijoles Rancheros — 22.71g
- Hummus with Red Chilli and Cumin recipes — 22.46g
- Artichoke Hummus recipes — 16.24g
- Falling for Pumpkin recipes — 15.80g

All satisfy Vegan + Lunch/Dinner + protein >=15g.

### Recall@K evaluation
Populate `data/eval/golden_questions.csv`, then:

```bash
.venv/bin/python evals/retrieval_eval.py
```

### End-to-end UI
Run Streamlit and verify generated answers, `[R#]` citations, source URLs, and similarity scores.

## Known limitations

- The Streamlit baseline uses dense retrieval unless metadata filters are explicitly supplied by code.
- Natural-language constraints are not automatically converted into Pinecone filters.
- Dense retrieval alone does not guarantee dietary, meal-type, allergen, or numeric constraints.
- LLM generation can introduce unsupported claims; explicit source evidence/metadata should be used for safety-related dietary claims.
- The 15g threshold is an experimental application rule, not a general nutritional definition.

Automatic constraint interpretation and tool selection are deferred to the agentic follow-on project.
