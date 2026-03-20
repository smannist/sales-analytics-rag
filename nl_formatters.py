import pandas as pd

from nl_templates import (
    MONTHLY_SUMMARY,
    TOP_CATEGORY_SUMMARY,
    TRANSACTION_DESCRIPTION,
)


def transaction_nl(
    row: pd.Series,
    template: str = TRANSACTION_DESCRIPTION,
) -> str:
    """Converts a transaction row to a natural language description.

    Args:
        row: A pandas row containing transaction data.
        template: A format string template for the natural language description.

    Returns:
        A natural language description of the transaction.
    """
    return template.format_map(
        {
            **row,
            "Sales": f"${row['Sales']:.2f}",
            "Profit": f"${row['Profit']:.2f}",
            "Discount": f"{row['Discount'] * 100:.0f}%",
        }
    )


def monthly_sales_nl(
    row: pd.Series,
    template: str = MONTHLY_SUMMARY,
) -> str:
    """Converts a monthly sales row to a natural language description.

    Args:
        row: A pandas row containing aggregated monthly sales.
        template: A format string template for the natural language description.

    Returns:
        A natural language description of the monthly sales.
    """
    return template.format_map(
        {
            **row,
            "Month": int(row["Month"]),
            "Year": int(row["Year"]),
            "Total_Sales": f"${row['Total_Sales']:,.2f}",
            "Total_Profit": f"${row['Total_Profit']:,.2f}",
            "Avg_Discount": f"{row['Avg_Discount'] * 100:.0f}%",
        }
    )


def top_category_nl(
    df: pd.DataFrame,
    template: str = TOP_CATEGORY_SUMMARY,
) -> str:
    """Converts the category revenue dataframe to one natural language summary.

    Args:
        df: A pandas dataframe containing category revenue data.
        template: A format string template for the natural language description.

    Returns:
        A natural language summary of the ranked category revenue data.
    """
    return template.format_map(
        {
            "Category_Rankings": "\n".join(
                f"{rank}. {row.Category}: ${row.Total_Sales:,.2f} in total revenue."
                for rank, row in enumerate(df.itertuples(index=False), start=1)
            )
        }
    )
