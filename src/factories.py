import pandas as pd

from aggregates import calculate_monthly_sales, calculate_top_categories
from document import Document
from metadata import (
    MONTHLY_METADATA_FIELDS,
    TRANSACTION_METADATA_FIELDS,
    extract_metadata,
)
from register import register


@register
def transactions(df: pd.DataFrame) -> Document:
    """Builds a Document from all transaction rows.

    Args:
        df: The pandas dataframe.

    Returns:
        A Document containing natural language descriptions of each transaction.
    """
    return Document(
        name="transactions",
        data=[
            f"On {row['Order Date']}, a {row['Segment']} customer based in "
            f"{row['City']}, {row['State']} ({row['Region']} region) "
            f"purchased {row['Product Name']} "
            f"({row['Sub-Category']}, {row['Category']}). "
            f"{row['Quantity']} unit(s) sold at ${row['Sales']:.2f} "
            f"with a {row['Discount'] * 100:.0f}% discount, "
            f"yielding a profit of ${row['Profit']:.2f}. "
            f"Shipped via {row['Ship Mode']}."
            for _, row in df.iterrows()
        ],
        metadata=df.apply(
            extract_metadata,
            args=(TRANSACTION_METADATA_FIELDS,),
            axis=1,
        ).tolist(),
    )


@register
def monthly_sales(df: pd.DataFrame) -> Document:
    """Builds a Document from monthly sales aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A Document containing monthly sales summaries.
    """
    monthly_sales_df = calculate_monthly_sales(df)
    return Document(
        name="monthly_sales",
        data=[
            f"In {int(row['Month'])}/{int(row['Year'])}, total sales were "
            f"${row['Total_Sales']:,.2f} and total profit was "
            f"${row['Total_Profit']:,.2f}. "
            f"The average discount that month was "
            f"{row['Avg_Discount'] * 100:.0f}%."
            for _, row in monthly_sales_df.iterrows()
        ],
        metadata=monthly_sales_df.apply(
            extract_metadata,
            args=(MONTHLY_METADATA_FIELDS,),
            axis=1,
        ).tolist(),
    )


@register
def top_categories(df: pd.DataFrame) -> Document:
    """Builds a Document from top category aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A Document containing a category revenue summary.
    """
    return Document(
        name="top_categories",
        data=[
            (
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
            )
        ],
    )
