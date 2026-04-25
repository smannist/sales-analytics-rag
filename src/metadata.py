import pandas as pd

type Metadata = dict[str, str | float | int]


TRANSACTION_METADATA_FIELDS: list[str] = [
    "Order Date",
    "Category",
    "Sub-Category",
    "Region",
    "State",
    "City",
    "Sales",
    "Profit",
]


MONTHLY_METADATA_FIELDS: list[str] = [
    "Month",
    "Year",
    "Total_Sales",
    "Total_Profit",
    "Avg_Discount",
]


MONTHLY_TOTAL_METADATA_FIELDS: list[str] = [
    "Month",
    "Total_Sales",
    "Total_Profit",
    "Avg_Discount",
]


YEARLY_METADATA_FIELDS: list[str] = [
    "Year",
    "Total_Sales",
    "Total_Profit",
    "Avg_Discount",
    "Total_Quantity",
]


CATEGORY_METADATA_FIELDS: list[str] = [
    "Category",
    "Total_Sales",
    "Total_Profit",
    "Avg_Discount",
    "Total_Quantity",
]


SUB_CATEGORY_METADATA_FIELDS: list[str] = [
    "Category",
    "Sub-Category",
    "Total_Sales",
    "Total_Profit",
    "Avg_Discount",
    "Total_Quantity",
    "Profit_Margin",
]


REGIONAL_METADATA_FIELDS: list[str] = [
    "Region",
    "Total_Sales",
    "Total_Profit",
    "Avg_Discount",
    "Total_Quantity",
]


DISCOUNTED_PRODUCT_METADATA_FIELDS: list[str] = [
    "Product Name",
    "Category",
    "Sub-Category",
    "Order_Count",
    "Discounted_Count",
    "Discount_Rate",
    "Avg_Discount",
]


def extract_metadata(
    row: pd.Series,
    fields: list[str],
) -> Metadata:
    """Extracts selected metadata fields from a row.

    Args:
        row: A pandas row containing data.
        fields: List of columns to include in the metadata dictionary.

    Returns:
        A dictionary including the metadata fields.
    """
    return {
        field: row[field] for field in fields if field in row
    }
