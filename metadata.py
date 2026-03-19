import pandas as pd

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


def extract_metadata(
    row: pd.Series,
    fields: list[str],
) -> dict[str, str | float | int]:
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
