# Meal Plan RAG Assistant — Assignment 1

A Streamlit RAG application for asking meal-selection and nutrition questions over a recipe corpus.

## Architecture
CSV recipe corpus → one LangChain Document per recipe → local SentenceTransformer embeddings → Pinecone dense retrieval (Top-K=5) → LangGraph evidence gate → local Ollama LLM → grounded answer + recipe citations.

The baseline intentionally keeps **one recipe = one chunk**. It does not use hybrid search, reranking, or metadata filters yet; those are evaluation-driven iterations.

## Dataset schema expected
`recipe_name, source, url, servings, calories, total_weight_g, image_url, diet_labels, health_labels, cautions, cuisine_type, meal_type, dish_type, ingredient_lines, ingredients, total_nutrients, daily_values, digest`

`total_nutrients` is expected to contain Edamam-style nutrient keys such as `PROCNT`, `FAT`, and `CHOCDF`.

## 1. Create environment (macOS)
```bash
cd meal-plan-rag-assignment1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Install Ollama and pull a local generator
Install Ollama, then:
```bash
ollama pull llama3.2:3b
ollama serve
```
If Ollama is already running, `ollama serve` is unnecessary.

## 3. Configure Pinecone
Create a Pinecone account/project and get an API key.

```bash
cp .env.example .env
```
Edit `.env` and set `PINECONE_API_KEY`.

The first ingestion creates a 384-dimensional cosine index because `all-MiniLM-L6-v2` outputs 384-dimensional embeddings.

## 4. Add your corpus
Put your real CSV at:
`data/raw/recipes-with-nutrition.csv`

A tiny synthetic `sample_recipes.csv` is included only for smoke testing.

## 5. Ingest
First test with the sample:
```bash
python -m src.ingest --csv data/raw/sample_recipes.csv
```

Then ingest the real corpus. Start small:
```bash
python -m src.ingest --csv data/raw/recipes-with-nutrition.csv --limit 300
```

## 6. Run Streamlit
```bash
streamlit run app/streamlit_app.py
```
Open the local URL Streamlit prints.

Try:
- Find a high-protein vegan dinner.
- Which retrieved dinner has the most protein per serving?
- Find a vegetarian breakfast.
- Is the tofu stir fry safe for someone avoiding soy?

The last question is intentionally useful: the app should avoid claiming allergen safety when the corpus does not explicitly support it.

## 7. Retrieval evaluation
After choosing real recipes, fill:
`data/eval/golden_questions.csv`

Format:
```csv
question,expected_recipe
How much protein is in Lentil Chili?,Lentil Chili
```

Run:
```bash
python evals/retrieval_eval.py
```

This reports a simple Recall@K baseline.

## Assignment-ready experiments
1. Whole-recipe chunks (baseline) vs fixed-size chunks.
2. Top-K 3 vs 5 vs 10.
3. Dense-only baseline vs hybrid retrieval.
4. Prompt v1 vs a stricter grounding/refusal prompt.
5. Later: metadata filtering for exact constraints such as meal type and numeric protein thresholds.

## Important limitation
This is a learning RAG application, not an allergy-safety system. Retrieval cannot reliably infer hidden ingredient relationships such as tofu → soy unless that relationship is represented in the corpus or a later validation tool/ontology.

## Project documentation notes
Document:
- why recipes with nutrition-per-serving were chosen;
- why one recipe is initially one chunk;
- why the same embedding model is used for corpus and query;
- retrieval vs generation failures;
- tofu/soy limitation;
- evaluation results before/after each iteration.
