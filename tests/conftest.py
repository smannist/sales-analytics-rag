import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import BaseMessage

from llm import (
    determine_retrieval_plan,
    generate_answer,
    generate_followup_answer,
    retrieve,
)
from vectorstore import get_vectorstore


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
    plan = determine_retrieval_plan(question, history=[])
    documents = retrieve(vectorstore, plan)
    actual_output = generate_answer(question, documents, history=[])

    correctness = GEval(
        name="Correctness",
        criteria=(
            f"Determine if the 'actual output' {criteria_detail}, "
            "consistent with the 'expected output'. "
            "The actual output will typically include a brief analysis paragraph after the "
            "bulleted list. ONLY judge the analysis on the correctness of what it DOES say. "
            "Verify that every number, percentage, derivation, and comparison present in the "
            "analysis is factually accurate and internally consistent with the figures in the "
            "bullets. Penalize fabricated numbers, prose percentages that disagree with their "
            "derivations, and unsupported qualitative claims (e.g. 'doubled', 'accelerated' when "
            "the data does not actually show it). "
            "Do not penalize the analysis for omitting specific observations, ratios, framings, "
            "or topics that appear in the 'expected output'. The expected analysis is "
            "illustrative, not a checklist: any number of grounded observations are acceptable as "
            "long as they are correct."
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
        retrieval_context=[doc.page_content for doc in documents],
    )

    assert_test(test_case, [correctness])


def _run_followup_test(
    history: list[BaseMessage],
    question: str,
    expected_output: str,
    criteria_detail: str,
) -> None:
    actual_output = generate_followup_answer(question, history=history)

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
