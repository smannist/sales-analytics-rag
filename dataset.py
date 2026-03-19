from pathlib import Path

import kagglehub
import pandas as pd

TRANSACTION_DESCRIPTION: str = (
    "On {Order Date}, a {Segment} customer based in {City}, {State} ({Region} region) "
    "purchased {Product Name} ({Sub-Category}, {Category}). "
    "{Quantity} unit(s) sold at {Sales} with a {Discount} discount, "
    "yielding a profit of {Profit}. Shipped via {Ship Mode}."
)


METADATA_FIELDS: list[str] = [
    "Order Date",
    "Category",
    "Sub-Category",
    "Region",
    "State",
    "City",
    "Sales",
    "Profit",
]


def download_and_save(
    dataset_location: str,
    dataset_name: str,
    output_path: str,
    encoding: str = "utf-8"
) -> None:
    """Downloads a dataset from Kaggle and saves it as a local CSV file.

    Args:
        dataset_location: The location of the dataset on Kaggle.
        dataset_name: The name of the dataset file.
        output_path: The local path to save the CSV file to.
        encoding: The encoding of the dataset (optional, default is utf-8).

    Raises:
        RuntimeError: If the dataset download or save fails.
    """
    try:
        base_path = kagglehub.dataset_download(dataset_location)
        df = pd.read_csv(Path(base_path) / dataset_name, encoding=encoding)
        df.to_csv(output_path, index=False)
        print(f"Finished. Dataset saved to {output_path}.")
    except Exception as e:
        msg = f"Error downloading dataset: {e}"
        raise RuntimeError(msg) from e


def extract_metadata(
    row: pd.Series,
    fields: list[str] = METADATA_FIELDS
) -> dict[str, str | float | int]:
    """Extracts selected metadata fields from a transaction row.

    Args:
        row: A pandas row containing transaction data.
        fields: List of columns to include in the metadata dictionary.

    Returns:
        A dictionary including the metadata fields.
    """
    return {
        field: row[field] for field in fields if field in row
    }


def transaction_to_nl(
    row: pd.Series,
    template: str = TRANSACTION_DESCRIPTION
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
