import calendar

import pandas as pd
from langchain_core.documents import Document

from aggregates import (
    calculate_category_aggregates,
    calculate_monthly_sales,
    calculate_monthly_totals,
    calculate_regional_sales,
    calculate_sub_category_margins,
    calculate_top_discounted_products,
    calculate_yearly_sales,
)
from metadata import (
    CATEGORY_METADATA_FIELDS,
    DISCOUNTED_PRODUCT_METADATA_FIELDS,
    MONTHLY_METADATA_FIELDS,
    MONTHLY_TOTAL_METADATA_FIELDS,
    REGIONAL_METADATA_FIELDS,
    SUB_CATEGORY_METADATA_FIELDS,
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
                f"{row['Order Date']}: {row['Segment']} customer in "
                f"{row['City']}, {row['State']} ({row['Region']}). "
                f"Bought {row['Product Name']} "
                f"({row['Sub-Category']}, {row['Category']}). "
                f"Qty {row['Quantity']}, ${row['Sales']:.2f} sales, "
                f"{row['Discount'] * 100:.0f}% discount, "
                f"${row['Profit']:.2f} profit. "
                f"Via {row['Ship Mode']}."
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
                f"{calendar.month_name[int(row['Month'])]} had "
                f"total sales of ${row['Total_Sales']:,.2f}."
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
    """Builds Documents from per-category sales aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents, one per category, summarising sales,
        profit, quantity, and average discount.
    """
    return [
        Document(
            page_content=(
                f"Category {row['Category']} had total sales of "
                f"${row['Total_Sales']:,.2f}, total profit of "
                f"${row['Total_Profit']:,.2f}, "
                f"{int(row['Total_Quantity'])} units sold, "
                f"and an average discount of "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata={
                **extract_metadata(row, CATEGORY_METADATA_FIELDS),
                "doc_type": "category",
            },
        )
        for _, row in calculate_category_aggregates(df).iterrows()
    ]


@document_factory
def top_sub_categories(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from per-sub-category aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents, one per sub-category, with profit margin,
        total sales, profit, quantity, and average discount.
    """
    return [
        Document(
            page_content=(
                f"Sub-category {row['Sub-Category']} "
                f"({row['Category']}) had a profit margin of "
                f"{row['Profit_Margin'] * 100:.2f}%."
            ),
            metadata={
                **extract_metadata(row, SUB_CATEGORY_METADATA_FIELDS),
                "doc_type": "sub_category",
            },
        )
        for _, row in calculate_sub_category_margins(df).iterrows()
    ]


@document_factory
def regional_sales(df: pd.DataFrame) -> list[Document]:
    """Builds Documents from per-region sales aggregates.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of Documents, one per region, with total sales, profit,
        quantity, and average discount.
    """
    return [
        Document(
            page_content=(
                f"Region {row['Region']} had total sales of "
                f"${row['Total_Sales']:,.2f}, total profit of "
                f"${row['Total_Profit']:,.2f}, "
                f"{int(row['Total_Quantity'])} units sold, "
                f"and an average discount of "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata={
                **extract_metadata(row, REGIONAL_METADATA_FIELDS),
                "doc_type": "regional",
            },
        )
        for _, row in calculate_regional_sales(df).iterrows()
    ]


@document_factory
def top_discounted_products(df: pd.DataFrame) -> list[Document]:
    """Builds Documents for the top 10 most frequently discounted products.

    Args:
        df: The pandas dataframe.

    Returns:
        A list of 10 Documents, one per product.
    """
    return [
        Document(
            page_content=(
                f"Product {row['Product Name']} "
                f"({row['Sub-Category']}, {row['Category']}) was "
                f"discounted in {int(row['Discounted_Count'])} of "
                f"{int(row['Order_Count'])} orders "
                f"({row['Discount_Rate'] * 100:.0f}% discount rate), "
                f"with an average discount of "
                f"{row['Avg_Discount'] * 100:.0f}%."
            ),
            metadata={
                **extract_metadata(row, DISCOUNTED_PRODUCT_METADATA_FIELDS),
                "doc_type": "discounted_product",
            },
        )
        for _, row in calculate_top_discounted_products(df).iterrows()
    ]
