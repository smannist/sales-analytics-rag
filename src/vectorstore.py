import chromadb
import pandas as pd
from langchain_chroma import Chroma

import factories as _  # noqa: F401 -- current decorator pattern requires importing the factories too
from config import CliMessage, VectorStoreConfig
from console import console
from registry import DOCUMENT_FACTORY_REGISTRY


def populate_vectorstore(vectorstore: Chroma, df: pd.DataFrame) -> None:
    """Inserts documents into the vectorstore.

    Args:
        vectorstore: The Chroma vectorstore.
        df: The pandas DataFrame.
    """
    all_docs = [doc for fn in DOCUMENT_FACTORY_REGISTRY for doc in fn(df)]

    with console.status(CliMessage.INSERTING):
        batch_size = 500
        for i in range(0, len(all_docs), batch_size):
            vectorstore.add_documents(all_docs[i : i + batch_size])
            console.print(
                f"  {min(i + batch_size, len(all_docs))}/{len(all_docs)} inserted.",
                style="dim",
            )

    console.print(CliMessage.INSERTED)


def get_vectorstore(df: pd.DataFrame) -> Chroma:
    """Returns a Chroma vectorstore, populating it on first run if needed.

    Args:
        df: The pandas DataFrame.

    Returns:
        The Chroma vectorstore instance.
    """
    client = chromadb.PersistentClient(path=VectorStoreConfig.PATH)

    vectorstore = Chroma(
        client=client,
        collection_name=VectorStoreConfig.COLLECTION_NAME
    )

    if client.get_collection(VectorStoreConfig.COLLECTION_NAME).count() == 0:
        populate_vectorstore(vectorstore, df)
    else:
        console.print(CliMessage.ALREADY_POPULATED)

    return vectorstore
