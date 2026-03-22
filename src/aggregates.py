import pandas as pd


def calculate_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates monthly sales data for each month/year.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe containing the sales data for each month/year.
    """
    return (
        df
        .assign(
            Month=pd.to_datetime(df["Order Date"]).dt.month,
            Year=pd.to_datetime(df["Order Date"]).dt.year,
        )
        .groupby(["Year", "Month"])
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
        )
        .reset_index()
    )


def calculate_top_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the top categories by total revenue.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe containing categories ordered by total revenue.
    """
    return (
        df
        .groupby("Category")
        .agg(Total_Sales=("Sales", "sum"))
        .reset_index()
        .sort_values("Total_Sales", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
