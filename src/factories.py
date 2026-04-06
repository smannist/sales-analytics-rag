import calendar

import pandas as pd
from langchain_core.documents import Document

from aggregates import (
    calculate_monthly_sales,
    calculate_monthly_totals,
    calculate_regional_sales,
    calculate_top_categories,
    calculate_top_discounted_products,
    calculate_top_sub_categories,
    calculate_yearly_sales,
)
from metadata import (
    MONTHLY_METADATA_FIELDS,
    MONTHLY_TOTAL_METADATA_FIELDS,
    TRANSACTION_METADATA_FIELDS,
    YEARLY_METADATA_FIELDS,
    extract_metadata,
)
from registry import document_factory


@document_factory
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
            metadata={
                **extract_metadata(row, TRANSACTION_METADATA_FIELDS),
                "doc_type": "transaction",
            },
        )
        for _, row in df.iterrows()
    ]


@document_factory
def monthly_sales(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from monthly sales aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents containing monthly sales summaries.
    """
    return [
        Document(
            page_content=(
                f"In {int(row['Month'])}/{int(row['Year'])}, total sales were "
                f"${row['Total_Sales']:,.2f} and total profit was "
                f"${row['Total_Profit']:,.2f}. "
                f"The average discount that month was "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata={
                **extract_metadata(row, MONTHLY_METADATA_FIELDS),
                "doc_type": "monthly",
            },
        )
        for _, row in calculate_monthly_sales(df).iterrows()
    ]


@document_factory
def monthly_totals(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from monthly totals aggregated across all years.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of 12 Documents, one per calendar month, summarising
        sales across all years.
    """
    return [
        Document(
            page_content=(
                f"Across all years, {calendar.month_name[int(row['Month'])]} "
                f"(month {int(row['Month'])}) had total sales of "
                f"${row['Total_Sales']:,.2f} and total profit of "
                f"${row['Total_Profit']:,.2f}. "
                f"The average discount was "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata={
                **extract_metadata(row, MONTHLY_TOTAL_METADATA_FIELDS),
                "doc_type": "monthly_total",
            },
        )
        for _, row in calculate_monthly_totals(df).iterrows()
    ]


@document_factory
def yearly_sales(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from yearly sales aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents containing yearly sales summaries.
    """
    return [
        Document(
            page_content=(
                f"In {int(row['Year'])}, total sales were "
                f"${row['Total_Sales']:,.2f} and total profit was "
                f"${row['Total_Profit']:,.2f}. "
                f"A total of {int(row['Total_Quantity'])} units were sold "
                f"with an average discount of "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata={
                **extract_metadata(row, YEARLY_METADATA_FIELDS),
                "doc_type": "yearly",
            },
        )
        for _, row in calculate_yearly_sales(df).iterrows()
    ]


@document_factory
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
                    f"{idx}. {row['Category']}: "
                    f"${row['Total_Sales']:,.2f} in total revenue."
                    for idx, (_, row) in enumerate(
                        calculate_top_categories(df).iterrows(),
                        start=1,
                    )
                )
            ),
            metadata={"doc_type": "category"},
        )
    ]


@document_factory
def top_sub_categories(df: pd.DataFrame) -> list[Document]:
    """Builds a Document from sub-category profit margin aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        Sub-categories ranked by profit margin.
    """
    return [
        Document(
            page_content=(
                "Sub-categories by profit margin (profit / sales), "
                "ranked from highest to lowest:\n"
                + "\n".join(
                    f"{idx}. {row['Sub-Category']} ({row['Category']}): "
                    f"{row['Profit_Margin'] * 100:.2f}% margin, "
                    f"${row['Total_Sales']:,.2f} in total sales, "
                    f"${row['Total_Profit']:,.2f} in total profit."
                    for idx, (_, row) in enumerate(
                        calculate_top_sub_categories(df).iterrows(),
                        start=1,
                    )
                )
            ),
            metadata={"doc_type": "sub_category"},
        )
    ]


@document_factory
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
                    f"{idx}. {row['Region']}: "
                    f"${row['Total_Sales']:,.2f} in total sales, "
                    f"${row['Total_Profit']:,.2f} in total profit, "
                    f"{int(row['Total_Quantity'])} units sold, "
                    f"average discount of {row['Avg_Discount'] * 100:.0f}%."
                    for idx, (_, row) in enumerate(
                        calculate_regional_sales(df).iterrows(),
                        start=1,
                    )
                )
            ),
            metadata={"doc_type": "regional"},
        )
    ]


@document_factory
def top_discounted_products(df: pd.DataFrame) -> list[Document]:
    """Builds a Document listing the top 10 most frequently discounted products.

    Args:
        df: The pandas dataframe.

    Returns:
        A list containing a single Document with the top 10 products
        most frequently sold at a discount.
    """
    return [
        Document(
            page_content=(
                "Top 10 products most frequently sold at a discount:\n"
                + "\n".join(
                    f"{idx}. {row['Product Name']} "
                    f"({row['Sub-Category']}, {row['Category']}): "
                    f"discounted in {int(row['Discounted_Count'])} of "
                    f"{int(row['Order_Count'])} orders "
                    f"({row['Discount_Rate'] * 100:.0f}% discount rate), "
                    f"average discount {row['Avg_Discount'] * 100:.0f}%."
                    for idx, (_, row) in enumerate(
                        calculate_top_discounted_products(df).iterrows(),
                        start=1,
                    )
                )
            ),
            metadata={"doc_type": "discounted_product"},
        )
    ]
