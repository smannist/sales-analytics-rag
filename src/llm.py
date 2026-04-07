from collections import defaultdict
from enum import StrEnum
from typing import Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from config import OpenAIConfig
from utils import load_file

load_dotenv(override=True)


model = ChatOpenAI(model=OpenAIConfig.MODEL)


PROMPT_RETRIEVAL_STRATEGY = load_file("prompts/retrieval_strategy.txt")
PROMPT_DATA_ENRICHER = load_file("prompts/data_enricher.txt")
PROMPT_FOLLOWUP_ANSWER = load_file("prompts/followup_answer.txt")


class MetadataFilter(BaseModel):
    """A single metadata filter condition."""
    field: str = Field(
        description="Metadata field name, e.g. 'Category', 'Region', 'State', 'Year'"
    )
    operator: Literal["$eq", "$gt", "$gte", "$lt", "$lte"] = Field(
        description=(
            "Comparison operator. To express OR across multiple values of "
            "the same field (e.g. Year 2014 OR 2015), emit multiple $eq "
            "filters with the same field — they will be combined into an "
            "$in clause by the caller. Do NOT use $or here."
        ),
        default="$eq",
    )
    value: str | int | float = Field(
        description="The value to filter by"
    )


class AnswerStrategy(StrEnum):
    """Determinate answer strategy: filters, just similarity or follow up."""
    METADATA_FILTER = "metadata_filter"
    SIMILARITY = "similarity"
    FOLLOW_UP = "follow_up"


class RetrievalPlan(BaseModel):
    """Full retrieval plan: strategy, filters, etc."""
    strategy: AnswerStrategy = Field(
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
        ge=1,
        description=(
            "Number of documents to retrieve. Must be at least 1. "
            "Infer from the question: e.g., 4 for a 4-year trend, "
            "12 for monthly breakdown, 1 for a single-entity lookup. "
            "Default to 3 when unclear."
        )
    )


@traceable(run_type="llm", name="Retrieval plan")
def determine_retrieval_plan(
        question: str,
        history: list[BaseMessage],
) -> RetrievalPlan:
    """Ask the LLM to determine the user's question into a retrieval plan.

    Args:
        question: The user's query string.
        history: Prior chat history to provide context for follow-ups.

    Returns:
        The full retrieval plan.
    """
    try:
        structured_model = model.with_structured_output(RetrievalPlan)
        return structured_model.invoke(  # type: ignore[return-value]
            [
                SystemMessage(
                    content=PROMPT_RETRIEVAL_STRATEGY
                ),
                *history,
                HumanMessage(
                    content=question
                ),
            ]
        )
    except Exception:  # just in case bad things happen
        return RetrievalPlan(
            strategy=AnswerStrategy.SIMILARITY,
            filters=[],
            user_query=question,
        )


@traceable(run_type="llm", name="Enricher Answer")
def generate_answer(
        question: str,
        documents: list[Document],
        history: list[BaseMessage],
) -> str:
    """Generates an enriched answer from the retrieved documents.

    Args:
        question: The user's query string.
        documents: Documents from the vector store.
        history: Prior chat history to provide context for follow-ups.

    Returns:
        LLM-generated answer based on the documents.
    """
    response = model.invoke(
        [
            SystemMessage(
                content=PROMPT_DATA_ENRICHER.format(
                    context="\n\n".join(
                        doc.page_content for doc in documents
                    )
                )
            ),
            *history,
            HumanMessage(
                content=question
            ),
        ]
    )
    return response.content


@traceable(run_type="llm", name="Follow-up Answer")
def generate_followup_answer(
        question: str,
        history: list[BaseMessage],
) -> str:
    """Answer a follow-up question purely from prior conversation context.

    Args:
        question: The user's follow-up query.
        history: Prior chat history containing the answer context.

    Returns:
        LLM-generated answer based on prior conversation.
    """
    response = model.invoke(
        [
            SystemMessage(
                content=PROMPT_FOLLOWUP_ANSWER
            ),
            *history,
            HumanMessage(
                content=question
            ),
        ]
    )
    return response.content


@traceable(run_type="retriever", name="Vector Retrieval")
def retrieve(vectorstore: Chroma, plan: RetrievalPlan) -> list[Document]:
    """Execute the retrieval strategy defined in the retrieval plan.

    Args:
        vectorstore: The Chroma vectorstore.
        plan: The retrieval plan containing strategy, filters, query, and k.

    Returns:
        The retrieved documents from vectorstore.
    """
    where_clause = None
    if plan.strategy == AnswerStrategy.METADATA_FILTER:
        where_clause = filters_to_where_clause(plan.filters)
    return vectorstore.similarity_search(
        plan.user_query,
        k=plan.k,
        filter=where_clause
    )


# langchain has built-in QueryRetriever (self, multi),
# but afaik it's prompt templates cannot be modified
# so this might be better for our use case??
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
