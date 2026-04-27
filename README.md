# sales-analytics-rag

Data Warehousing and Business Intelligence course project. Implements a Retrieval-Augmented Generation (RAG) system for analyzing real-world sales data using vector search and LLMs to generate insights into trends and patterns.

# Architecture

The app is fairly simple example of a RAG pipeline. It consists of retrieval planning and answer generation (with follow-up shortcut).

1. At retrieval planning stage, based on the user query, LLM decides which strategy to use (follow up vs no follow up) and which filters to apply for VectorDB search, e.g. meta filters and the number of documents to retrieve.
2. If the strategy is follow up, just use history data and answer based on that.
3. Otherwise, retrieve document(s) from the vector database, and use the documents as a context for the final output.

The app can provide answer to questions related to the superstore dataset, such as:

1. What is the sales trend over the 4-year period?
2. Which months show the highest sales? Is there seasonality?
3. Which product categories generate the most revenue?
4. What sub-categories have the highest profit margins?
5. Which regions have the best sales performance?
6. Compare Technology vs. Furniture sales
7. How does the West region compare to the East in terms of sales?
8. Any user follow up questions, e.g., "Why does November perform better than ...."

Note: the app is not really production ready, but more like a demoing app for demonstrating how RAG works and how LLMs can be used for simple analytics.

# Setup

### 1. Set environmental variables

Make sure that you have environmental variables set up.

.env.example provides a guide for this. But in short, you'll only need either a GROQ_API_KEY or OPENAI_API_KEY set in your .env file.

### 2. Ready virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

Either via uv (skip the first command if you already have uv installed):

```bash
pip install uv
```

```bash
uv sync
```

Or with plain pip via requirements.txt:

```bash
pip install -r requirements.txt
```

# Running the app

This assumes that you are running the command from the root folder.

```bash
python3 src/app.py run
```

# Note about the sections below

Depending on what you are using (uv vs pip) use `uv run` or `python3` prefix in front of the commands.

# Linting

The app uses Ruff, which you can run by doing:

```bash
ruff check
```

# Type check

Ty is used for type checking, run this from root:

```bash
ty check --error all ./src
```

# Testing

The app contains a few tests, it's recommended to run them separately:

E.g. LLM evaluation uses LLM-as-a-judge style, with deepevals. Running just the enricher tests works as follows:

```bash
deepeval test run tests/evals/test_eval_rag_enricher.py
```

these can also be explored more deeply at DeepEval website, but requires an API KEY.

Unit tests use pytest, and runs with:

```bash
pytest tests/units
```
