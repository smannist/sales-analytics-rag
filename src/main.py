from pathlib import Path

import chromadb
import kagglehub
import pandas as pd
from langchain_chroma import Chroma

import factories as _  # noqa: F401 -- current decorator pattern requires importing the factories too
from register import DOCUMENT_REGISTRY


def main() -> None:
    """Main function to run the program."""
    try:
        df = pd.read_csv("superstore.csv", encoding="ISO-8859-1")
        print("Dataset loaded from a local file.")
    except FileNotFoundError:
        print("Dataset not found. Downloading from Kaggle...")
        base_path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
        dataset_path = Path(base_path) / "Sample - Superstore.csv"
        df = pd.read_csv(dataset_path, encoding="ISO-8859-1")
        df.to_csv("superstore.csv", index=False)

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
