# sales-analytics-rag

Data Warehousing and Business Intelligence course project. Implements a Retrieval-Augmented Generation (RAG) system for analyzing real-world sales data using vector search and LLMs to generate insights into trends and patterns.

# Setup

1. Ready virtual environment

```Python
python3 -m venv .venv
```

2. Activate virtual environment

```Python
source .venv/bin/activate
```

3. Install uv (if you don't already have it)

```Python
pip install uv
```

4. Install dependencies

```python
uv sync
```

# Running the app

This assumes that you are running the command from the root folder.

```python
python3 src/app.py run
```

# Linting

The app uses Ruff, which you can run by doing:

```python
uv run ruff check
```

# Type check

To type check the source folder, run this from root:

```python
uv run ty check --error all ./src
```
