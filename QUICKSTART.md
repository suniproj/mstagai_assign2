# Fast submission path

1. Put `recipes-with-nutrition.csv` in `data/raw/`.

2. Prepare balanced 300-recipe corpus:
`.venv/bin/python -m src.prepare_corpus --input data/raw/recipes-with-nutrition.csv`

3. Delete old sample index:
`.venv/bin/python -m src.reset_index`

4. Ingest real corpus:
`.venv/bin/python -m src.ingest --csv data/processed/recipes_clean.csv`

5. Run UI:
`.venv/bin/python -m streamlit run app/streamlit_app.py`

Demo questions:
- Find me a high-protein vegan dinner.
- Find me a vegetarian breakfast.
- Find me a soy-free vegan meal.
- Ask a soy-safety question to demonstrate strict corpus grounding.

Baseline: one recipe/one chunk; all-MiniLM-L6-v2 384-d embeddings; Pinecone cosine dense Top-K=5; LangGraph evidence gate; Ollama Llama 3.2 3B; Streamlit; citations.
