import pandas as pd


def calculate_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates monthly sales data for each month/year.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe containing the sales data for each month/year.
    """
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Month"] = df["Order Date"].dt.month
    df["Year"] = df["Order Date"].dt.year

    return (
        df
        .groupby(["Year", "Month"])
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
        )
        .reset_index()
    )
