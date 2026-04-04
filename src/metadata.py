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


_AGGREGATE_METADATA_BASE: list[str] = ["Total_Sales", "Total_Profit", "Avg_Discount"]
MONTHLY_METADATA_FIELDS: list[str] = ["Month", "Year", *_AGGREGATE_METADATA_BASE]
MONTHLY_TOTAL_METADATA_FIELDS: list[str] = ["Month", *_AGGREGATE_METADATA_BASE]
YEARLY_METADATA_FIELDS: list[str] = [
    "Year",
    *_AGGREGATE_METADATA_BASE,
    "Total_Quantity"
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
