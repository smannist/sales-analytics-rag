from pathlib import Path

import chromadb
import pandas as pd

import factories as _  # noqa: F401 -- current decorator pattern requires importing the factories too
from kaggle_utils import download_and_save_csv
from register import DOCUMENT_REGISTRY


def main() -> None:
    """Main function to run the program."""
    if not Path("superstore.csv").exists():
        download_and_save_csv(
            dataset_location="vivek468/superstore-dataset-final",
            dataset_name="Sample - Superstore.csv",
            output_path="superstore.csv",
            encoding="ISO-8859-1",
        )
    else:
        print("Dataset already exists. Skipping download.")

    df = pd.read_csv("superstore.csv", encoding="ISO-8859-1")

    docs = [build(df) for build in DOCUMENT_REGISTRY]

    client = chromadb.PersistentClient(path="./chroma_db")

    for doc in docs:
        collection = client.get_or_create_collection(name=doc.name)
        if collection.count() == 0:
            total = len(doc.data)
            for i in range(0, total, 500):
                data_batch = doc.data[i:i + 500]
                metadata_batch = doc.metadata[i:i + 500] if doc.metadata else None
                id_batch = [str(j) for j in range(i, i + len(data_batch))]
                collection.add(
                    documents=data_batch,
                    metadatas=metadata_batch,
                    ids=id_batch,
                )
                inserted = min(i + 500, total)
                print(f"  [{doc.name}] {inserted}/{total} documents inserted.")
            print(f"Finished inserting {total} documents into '{doc.name}'.")
        else:
            print(f"Collection '{doc.name}' already populated. Skipping.")

    results = client.get_or_create_collection(
        name="monthly_sales"
    ).query(
        query_texts=["What is the sales trend over the 4-year period?"],
        n_results=10,
    )

    documents = results.get("documents")
    if documents:
        for doc in documents[0]:
            print(doc)


if __name__ == "__main__":
    main()
