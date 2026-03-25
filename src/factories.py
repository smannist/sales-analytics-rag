import pandas as pd
from langchain_core.documents import Document

from aggregates import (
    calculate_monthly_sales,
    calculate_regional_sales,
    calculate_top_categories,
)
from metadata import (
    MONTHLY_METADATA_FIELDS,
    TRANSACTION_METADATA_FIELDS,
    extract_metadata,
)
from register import register


@register
def transactions(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from all transaction rows.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents with natural language descriptions
        of each transaction.
    """
    return [
        Document(
            page_content=(
                f"On {row['Order Date']}, a {row['Segment']} customer based in "
                f"{row['City']}, {row['State']} ({row['Region']} region) "
                f"purchased {row['Product Name']} "
                f"({row['Sub-Category']}, {row['Category']}). "
                f"{row['Quantity']} unit(s) sold at ${row['Sales']:.2f} "
                f"with a {row['Discount'] * 100:.0f}% discount, "
                f"yielding a profit of ${row['Profit']:.2f}. "
                f"Shipped via {row['Ship Mode']}."
            ),
            metadata=extract_metadata(row, TRANSACTION_METADATA_FIELDS),
        )
        for _, row in df.iterrows()
    ]


@register
def monthly_sales(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from monthly sales aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents containing monthly sales summaries.
    """
    monthly_sales_df = calculate_monthly_sales(df)
    return [
        Document(
            page_content=(
                f"In {int(row['Month'])}/{int(row['Year'])}, total sales were "
                f"${row['Total_Sales']:,.2f} and total profit was "
                f"${row['Total_Profit']:,.2f}. "
                f"The average discount that month was "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata=extract_metadata(row, MONTHLY_METADATA_FIELDS),
        )
        for _, row in monthly_sales_df.iterrows()
    ]


@register
def top_categories(df: pd.DataFrame) -> list[Document]:
    """Builds a Document from top category aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list containing a single Document with category revenue summary.
    """
    return [
        Document(
            page_content=(
                "Top categories by total revenue, "
                "ranked from highest to lowest:\n"
                + "\n".join(
                    f"{idx}. {row.Category}: "
                    f"${row.Total_Sales:,.2f} in total revenue."
                    for idx, row in enumerate(
                        calculate_top_categories(df).itertuples(
                            index=False,
                        ),
                        start=1,
                    )
                )
            ),
        )
    ]


@register
def regional_sales(df: pd.DataFrame) -> list[Document]:
    """Builds a Document from regional sales performance aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list containing a single Document with sales performance by region.
    """
    return [
        Document(
            page_content=(
                "Sales performance by region, "
                "ranked from highest to lowest total sales:\n"
                + "\n".join(
                    f"{idx}. {row.Region}: "
                    f"${row.Total_Sales:,.2f} in total sales, "
                    f"${row.Total_Profit:,.2f} in total profit, "
                    f"{int(row.Total_Quantity)} units sold, "
                    f"average discount of {row.Avg_Discount * 100:.0f}%."
                    for idx, row in enumerate(
                        calculate_regional_sales(df).itertuples(index=False),
                        start=1,
                    )
                )
            ),
        )
    ]
