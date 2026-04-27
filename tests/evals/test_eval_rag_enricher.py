def test_best_region(run_rag_test) -> None:
    run_rag_test(
        question="Which regions have the best sales performance?",
        expected_output=(
            "The West region has the best sales performance, with $725,457.82 in total sales, "
            "ahead of East ($678,781.24), Central ($501,239.89), and South ($391,721.91). "
            "A brief analysis follows the list with a few grounded observations about the gaps "
            "between regions."
        ),
        criteria_detail=(
            "correctly identifies West as the top-performing region by total sales, "
            "with accurate sales figures for the listed regions"
        ),
    )


def test_seasonality(run_rag_test) -> None:
    run_rag_test(
        question="Which months show the highest sales? Is there seasonality?",
        expected_output=(
            "Correctly states that sales peak in November (~$352,461) and December (~$325,294). "
            "Other months such as September (~$307,650), March (~$205,005) and October (~$200,323) also appear. "
            "A brief analysis follows the list with a few grounded observations about the "
            "seasonality."
        ),
        criteria_detail=(
            "correctly identifies November and December as the top months, and includes the listed other months. "
            "Add some analysis on the seasonality. If there are any sales figures, they should show correct numbers and logic."
        ),
    )


def test_top_category(run_rag_test) -> None:
    run_rag_test(
        question="Which product categories generates the most revenue?",
        expected_output=(
            "Technology generates the most revenue at $836,154.03, ahead of Furniture "
            "($741,999.80) and Office Supplies ($719,047.03). "
            "A brief analysis follows the list with a few grounded observations about the gaps "
            "between categories."
        ),
        criteria_detail=(
            "correctly identifies Technology as the top-revenue category, "
            "with accurate revenue figures for the listed categories"
        ),
    )


def test_top_margin_subcategories(run_rag_test) -> None:
    run_rag_test(
        question="What sub-categories have the highest profit margins?",
        expected_output=(
            "Labels, Paper, Envelopes, Copiers and Fasteners have the highest profit margins, "
            "roughly 44%, 43%, 42%, 37% and 31% in that order. "
            "A brief analysis follows the list with a few grounded observations about the "
            "clustering and gaps between sub-categories."
        ),
        criteria_detail=(
            "correctly identifies the sub-categories with the highest profit margins, "
            "with margin percentages roughly matching"
        ),
    )


def test_west_vs_east_profit(run_rag_test) -> None:
    run_rag_test(
        question="How does the West region compare to the East in terms of profit?",
        expected_output=(
            "The West region is more profitable than the East: $108,418.45 in profit versus "
            "$91,522.78. "
            "A brief analysis follows the list with a few grounded observations about the gap "
            "between the two regions."
        ),
        criteria_detail=(
            "correctly states that the West region has higher profit than the East, "
            "with profit figures roughly matching"
        ),
    )


def test_sales_trend(run_rag_test) -> None:
    run_rag_test(
        question="What is the sales trend over the 4-year period?",
        expected_output=(
            "Yearly sales were $484,247.50 in 2014, $470,532.51 in 2015, $609,205.60 in 2016, "
            "and $733,215.26 in 2017. "
            "A brief analysis follows the list with a few grounded observations about the "
            "year-over-year changes and the overall trajectory."
        ),
        criteria_detail=(
            "correctly lists the sales figures for 2014, 2015, 2016, and 2017 in chronological "
            "order, with the values roughly matching"
        ),
    )


def test_frequently_discounted_products(run_rag_test) -> None:
    run_rag_test(
        question="Which products are frequently sold at a discount?",
        expected_output=(
            "The output names some discounted products. Each bullet shows the product name and its average discount percentage only, "
            "no discount rate. No strict ranking or ordering is necessary in this evaluation. "
            "A brief analysis follows the list with a few grounded observations about the "
            "products and the discount magnitudes."
        ),
        criteria_detail=(
            "correctly lists some discounted products (e.g. Staples, "
            "Staple envelope, Avery Non-Stick Binders, GBC Premium Transparent Covers, "
            "Easy-staple paper, and so on), shows exactly 5 bullets, each bullet displays the average "
            "discount percentage (not a 100% discount rate), and does not mention the actual discount rate"
        ),
    )
