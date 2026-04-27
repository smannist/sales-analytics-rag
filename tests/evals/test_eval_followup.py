from langchain_core.messages import AIMessage, HumanMessage

REGION_PROFIT_HISTORY = [
    HumanMessage(
        content="How does the West region compare to the East in terms of profit?"
    ),
    AIMessage(
        content=(
            "The West region is more profitable than the East: "
            "$108,418 in profit versus $91,523."
        ),
    ),
]


def test_interpretive_followup(run_followup_test) -> None:
    run_followup_test(
        history=REGION_PROFIT_HISTORY,
        question="Why do you think the West outperforms the East?",
        expected_output=(
            "West's higher profit may reflect stronger regional demand, better "
            "product mix, or lower operating costs, though these are hypotheses."
        ),
        criteria_detail="offers some sort of explanation for why West might outperform East",
    )
