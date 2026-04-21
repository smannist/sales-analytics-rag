def test_best_region(run_rag_test) -> None:
    run_rag_test(
        question="Which region has the best sales performance?",
        expected_output=(
            "The West region has the best sales performance, with $725,457.82 in total sales."
        ),
        criteria_detail="correctly identifies West as the top-performing region by total sales",
    )


def test_seasonality(run_rag_test) -> None:
    run_rag_test(
        question="Which months show the highest sales? Is there seasonality?",
        expected_output=(
            "Sales peak in November and December while January and February are the weakest months. "
            "Other months such as October and March are also listed. "
            "Seasonality is noticable e.g. November vs March, " 
            "could indicate Black Friday sales."
        ),
        criteria_detail=(
            "correctly identifies November, December as top months, "
            "highest-sales months and acknowledges the seasonality in some way"
        ),
    )


def test_top_category(run_rag_test) -> None:
    run_rag_test(
        question="Which product category generates the most revenue?",
        expected_output=(
            "Technology generates the most revenue at $836,154 ahead of the other two."
        ),
        criteria_detail="correctly identifies Technology as the top-revenue category",
    )


def test_top_margin_subcategories(run_rag_test) -> None:
    run_rag_test(
        question="What sub-categories have the highest profit margins?",
        expected_output=(
            "Labels, Paper, and Envelopes, Copiers and Fasteners have the "
            "highest profit margins, roughly 44%, 43%, 42%, 37% and 31% in that order."
        ),
        criteria_detail=(
            "correctly identifies the sub-categories with the highest profit margins"
        ),
    )


def test_west_vs_east_profit(run_rag_test) -> None:
    run_rag_test(
        question="How does the West region compare to the East in terms of profit?",
        expected_output=(
            "The West region is more profitable than the East: $108,418 in profit versus $91,523."
        ),
        criteria_detail=(
            "correctly states that the West region has higher profit than the East and figures match."
        ),
    )
