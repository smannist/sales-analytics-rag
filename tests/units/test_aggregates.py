import pandas as pd

from aggregates import calculate_regional_sales, calculate_sub_category_margins


def test_regional_sales_computed_correctly() -> None:
    df = pd.DataFrame({
        "Order Date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "Region": ["West", "East", "West", "East"],
        "Sales": [100.0, 50.0, 200.0, 25.0],
        "Profit": [10.0, 5.0, 20.0, 2.0],
        "Quantity": [1, 1, 2, 1],
        "Discount": [0.0, 0.1, 0.0, 0.2],
    })
    result = calculate_regional_sales(df)
    assert list(result["Region"]) == ["West", "East"]
    assert list(result["Total_Sales"]) == [300.0, 75.0]
    assert list(result["Total_Quantity"]) == [3, 2]


def test_sub_category_margins_computed_correctly() -> None:
    df = pd.DataFrame({
        "Category": ["Office Supplies", "Office Supplies", "Technology"],
        "Sub-Category": ["Labels", "Labels", "Phones"],
        "Sales": [100.0, 100.0, 500.0],
        "Profit": [40.0, 50.0, 50.0],
    })
    result = calculate_sub_category_margins(df)
    assert list(result["Sub-Category"]) == ["Labels", "Phones"]
    assert list(result["Profit_Margin"]) == [0.45, 0.1]
