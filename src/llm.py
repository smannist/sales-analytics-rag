import os
import warnings
from collections import defaultdict
from enum import StrEnum

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, Field

from config import OpenAIConfig
from utils import load_file

load_dotenv(override=True)


client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


PROMPT_RETRIEVAL_STRATEGY = load_file("prompts/retrieval_strategy.txt")
PROMPT_DATA_ENRICHER = load_file("prompts/data_enricher.txt")


class MetadataFilter(BaseModel):
    """A single metadata filter condition."""
    field: str = Field(
        description="Metadata field name, e.g. 'Category', 'Region', 'State', 'Year'"
    )
    operator: str = Field(
        description="Comparison operator: '$eq', '$gt', '$gte', '$lt', '$lte'",
        default="$eq"
    )
    value: str | int | float = Field(
        description="The value to filter by"
    )


class RetrievalStrategy(StrEnum):
    """Narrow down vectorstore retrieval strategy."""
    METADATA_FILTER = "metadata_filter"
    SIMILARITY = "similarity"


class RetrievalPlan(BaseModel):
    """Full retrieval plan: strategy, filters, etc."""
    strategy: RetrievalStrategy = Field(
        description="The retrieval strategy to use"
    )
    filters: list[MetadataFilter] = Field(
        default_factory=list,
        description="Metadata filters to apply."
        "Only populated when strategy is 'metadata_filter'."
    )
    user_query: str = Field(
        description="The query optimized for vector similarity retrieval"
    )
    k: int = Field(
        default=3,
        description=(
            "Number of documents to retrieve. Infer from the question: "
            "e.g., 4 for a 4-year trend, 12 for monthly breakdown, "
            "1 for a single-entity lookup. Default to 3 when unclear."
        )
    )


@traceable(run_type="llm", name="Query classifier")
def determine_retrieval_plan(question: str) -> RetrievalPlan:
    """Ask the LLM to determine the user's question into a retrieval plan.

    Args:
        question: The user's query string.

    Returns:
        The full retrieval plan.
    """
    with warnings.catch_warnings():  # parse gives false pydantic errors
        warnings.simplefilter("ignore", UserWarning)
        completion = client.chat.completions.parse(
            model=OpenAIConfig.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT_RETRIEVAL_STRATEGY
                },
                {
                    "role": "user",
                    "content": question
                },
            ],
            response_format=RetrievalPlan,
        )

    if not completion.choices[0].message.parsed:
        return RetrievalPlan(
            strategy=RetrievalStrategy.SIMILARITY,
            filters=[],
            user_query=question,
        )

    return completion.choices[0].message.parsed


@traceable(run_type="llm", name="Generated Answer")
def generate_answer(question: str, documents: list[Document]) -> str:
    """Generates an enriched answer from the retrieved documents.

    Args:
        question: The user's query string.
        documents: Documents from the vector store.

    Returns:
        LLM-generated answer based on the documents.
    """
    context = "\n\n".join(doc.page_content for doc in documents)
    response = client.chat.completions.create(
        model=OpenAIConfig.MODEL,
        messages=[
            {
                "role": "system",
                "content": PROMPT_DATA_ENRICHER.format(context=context)
            },
            {
                "role": "user",
                "content": question
            },
        ],
    )
    return response.choices[0].message.content or ""


@traceable(run_type="retriever", name="Vector Retrieval")
def retrieve(vectorstore: Chroma, plan: RetrievalPlan) -> list[Document]:
    """Execute the retrieval strategy defined in the retrieval plan.

    Results are sorted by Total_Sales descending when all retrieved
    documents carry that metadata field.

    Args:
        vectorstore: The Chroma vectorstore.
        plan: The retrieval plan containing strategy, filters, query, and k.

    Returns:
        The retrieved documents from vectorstore.
    """
    where_clause = None
    if plan.strategy == RetrievalStrategy.METADATA_FILTER:
        where_clause = filters_to_where_clause(plan.filters)
    docs = vectorstore.similarity_search(plan.user_query, k=plan.k, filter=where_clause)
    if docs and all("Total_Sales" in doc.metadata for doc in docs):
        docs.sort(key=lambda d: d.metadata["Total_Sales"], reverse=True)
    return docs


# langchain has built-in QueryRetriever (self, multi),
# but looks complex and I would need to learn more about it
# setup works, but maybe refactor this to use it
def filters_to_where_clause(filters: list[MetadataFilter]) -> dict | None:
    """Convert filters into a where clause for Chroma querying.

    Args:
        filters: The metadata filters.

    Returns:
        The constructed where clause, or None if no filters are provided.
    """
    if not filters:
        return None

    eq_groups: dict[str, list] = defaultdict(list)
    conditions: list[dict] = []

    for filt in filters:
        if filt.operator == "$eq":
            eq_groups[filt.field].append(filt.value)
        else:
            conditions.append({filt.field: {filt.operator: filt.value}})
    for field, values in eq_groups.items():
        conditions.append({field: {"$in": values}})

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]
