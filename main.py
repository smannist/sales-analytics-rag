from pathlib import Path

import chromadb
import pandas as pd

from dataset import download_and_save, extract_metadata, transaction_to_nl


# TODO: test program, will clean up and add rest later obvs
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

    documents = df.apply(transaction_to_nl, axis=1).tolist()
    metadatas = df.apply(extract_metadata, axis=1).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="superstore")

    if collection.count() == 0:
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            inserted = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=[str(j) for j in range(i, min(i + batch_size, len(documents)))],
            )
            print(f"Inserted {inserted}/{len(documents)} documents.")
        print("Finished inserting all documents.")
    else:
        print("Collection already exists. Skipping insertion.")

    results = collection.query(
        query_texts=["Which sales resulted in net loss in the West region?"],
        n_results=5,
        where={
            "$and": [
                {"Region": {"$eq": "West"}},
                {"Profit": {"$lt": 0.0}},
        ]
    },
    )

    for doc in results["documents"][0]:
        print(doc)


if __name__ == "__main__":
    main()
