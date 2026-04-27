import pytest
from deepeval import assert_test  # type: ignore[unresolved-import]
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import BaseMessage

from llm import (
    determine_retrieval_plan,
    generate_answer,
    generate_followup_answer,
    retrieve,
)
from utils import load_file
from vectorstore import get_vectorstore


RAG_CORRECTNESS_CRITERIA = load_file(__file__, "rag_correctness.txt")


@pytest.fixture
def run_rag_test():
    return _run_rag_test


@pytest.fixture
def run_followup_test():
    return _run_followup_test


def _run_rag_test(
        question: str,
        expected_output: str,
        criteria_detail: str
) -> None:
    vectorstore = get_vectorstore()
    plan = determine_retrieval_plan(question, history=[])             # type: ignore[invalid-argument-type]
    documents = retrieve(vectorstore, plan)                           # type: ignore[invalid-argument-type]
    actual_output = generate_answer(question, documents, history=[])  # type: ignore[invalid-argument-type]

    correctness = GEval(
        name="Correctness",
        criteria=RAG_CORRECTNESS_CRITERIA.format(criteria_detail=criteria_detail),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=[doc.page_content for doc in documents],
    )

    assert_test(test_case, [correctness])


def _run_followup_test(
    history: list[BaseMessage],
    question: str,
    expected_output: str,
    criteria_detail: str,
) -> None:
    actual_output = generate_followup_answer(question, history=history)  # type: ignore[invalid-argument-type]

    correctness = GEval(
        name="Correctness",
        criteria=(
            f"Determine if the 'actual output' {criteria_detail}, "
            "consistent with the 'expected output'."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
    )

    assert_test(test_case, [correctness])
