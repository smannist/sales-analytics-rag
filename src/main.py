from pathlib import Path

import chromadb
import pandas as pd
from langchain_chroma import Chroma

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
    client = chromadb.PersistentClient(path="./chroma_db")

    for name, build in DOCUMENT_REGISTRY.items():
        vectorstore = Chroma(
            client=client,
            collection_name=name,
        )
        if vectorstore._collection.count() == 0:  # noqa: SLF001
            docs = build(df)
            for i in range(0, len(docs), 500):
                vectorstore.add_documents(docs[i:i + 500])
                inserted = min(i + 500, len(docs))
                print(f"  [{name}] {inserted}/{len(docs)} inserted.")
            print(f"Finished inserting into '{name}'.")
        else:
            print(f"Collection '{name}' already populated. Skipping.")

    documents = Chroma(
        client=client,
        collection_name="top_categories",
    ).similarity_search(
        "Can you give me a summary of top categories by revenue?",
    )

    for doc in documents:
        print(doc.page_content)


if __name__ == "__main__":
    main()
