from pathlib import Path

import chromadb
import kagglehub
import pandas as pd
from langchain_chroma import Chroma

import factories as _  # noqa: F401 -- current decorator pattern requires importing the factories too
from registry import DOCUMENT_FACTORY_REGISTRY


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

    vectorstore = Chroma(
        client=client,
        collection_name="superstore"
    )

    if vectorstore._collection.count() == 0:
        all_docs = [doc for fn in DOCUMENT_FACTORY_REGISTRY for doc in fn(df)]

        for i in range(0, len(all_docs), 500):
            vectorstore.add_documents(all_docs[i:i + 500])
            print(f"{min(i + 500, len(all_docs))}/{len(all_docs)} inserted.")
        
        print("Finished inserting all documents.")
    else:
        print("Collection already populated. Skipping.")

    documents = vectorstore.similarity_search(
        "Can you give me a summary of top categories by revenue?",
        k=1
    )

    for doc in documents:
        print(doc.page_content)


if __name__ == "__main__":
    main()
