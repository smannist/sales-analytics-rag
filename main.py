from pathlib import Path

import chromadb
import pandas as pd

from aggregates import calculate_monthly_sales
from dataset import download_and_save
from metadata import MONTHLY_METADATA_FIELDS, extract_metadata
from nl_formatters import monthly_sales_nl


def main() -> None:
    """Main function to run the program."""
    if not Path("superstore.csv").exists():
        download_and_save(
            dataset_location="vivek468/superstore-dataset-final",
            dataset_name="Sample - Superstore.csv",
            output_path="superstore.csv",
            encoding="ISO-8859-1",
        )
    else:
        print("Dataset already exists. Skipping download.")

    df = pd.read_csv("superstore.csv", encoding="ISO-8859-1")
    monthly_sales = calculate_monthly_sales(df)

    summaries = monthly_sales.apply(monthly_sales_nl, axis=1).tolist()
    summary_metadatas = monthly_sales.apply(
        extract_metadata, args=(MONTHLY_METADATA_FIELDS,), axis=1
        ).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")

    summary_collection = client.get_or_create_collection(
        name="superstore_monthly_summaries"
    )

    if summary_collection.count() == 0:
        summary_collection.add(
            documents=summaries,
            metadatas=summary_metadatas,
            ids=[str(i) for i in range(len(summaries))],
        )
        print(f"Inserted {len(summaries)} monthly summaries.")
    else:
        print("Summary collection already exists. Skipping insertion.")

    summary_results = summary_collection.query(
        query_texts=["What is the sales trend over the 4-year period?"],
        n_results=10,
    )

    documents = summary_results.get("documents")
    if documents:
        for doc in documents[0]:
            print(doc)


if __name__ == "__main__":
    main()
