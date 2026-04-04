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


def calculate_monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates total sales per calendar month across all years.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe with monthly total sales.
    """
    return (
        df
        .assign(Month=pd.to_datetime(df["Order Date"]).dt.month)
        .groupby("Month")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
        )
        .reset_index()
        .sort_values(by="Month", kind="stable")
    )


def calculate_yearly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates yearly sales for each year.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe containing the sales data for each year.
    """
    return (
        df
        .assign(Year=pd.to_datetime(df["Order Date"]).dt.year)
        .groupby("Year")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Avg_Discount=("Discount", "mean"),
        )
        .reset_index()
        .sort_values(by="Year", kind="stable")
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
        .sort_values(
            by="Total_Sales",
            ascending=False,
            kind="stable"
        )
    )


def calculate_regional_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates sales performance by region.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe containing region's sales performance.
    """
    return (
        df
        .groupby("Region")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Avg_Discount=("Discount", "mean"),
        )
        .reset_index()
        .sort_values(
            by="Total_Sales",
            ascending=False,
            kind="stable"
        )
    )
