from collections.abc import Iterator

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import VectorDBConfig


def populate_vectorstore(
        vectorstore: Chroma,
        documents: list[Document]
) -> Iterator[tuple[int, int]]:
    """Inserts documents into the vectorstore, yielding progress after each batch.

    Args:
        vectorstore: The Chroma vectorstore.
        documents: A list of vectorstore documents.

    Yields:
        A tuple of (inserted_count, total_count) after each batch.
    """
    total = len(documents)
    for i in range(0, total, VectorDBConfig.INSERTION_BATCH_SIZE):
        vectorstore.add_documents(
            documents[i : i + VectorDBConfig.INSERTION_BATCH_SIZE]
        )
        yield min(i + VectorDBConfig.INSERTION_BATCH_SIZE, total), total


def get_vectorstore() -> Chroma:
    """Returns a Chroma vectorstore.

    Returns:
        The Chroma vectorstore instance.
    """
    return Chroma(
        client=chromadb.PersistentClient(path=VectorDBConfig.PATH),
        collection_name=VectorDBConfig.COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        ),
    )


def is_vectorstore_empty(vectorstore: Chroma) -> bool:
    """Return True if the vectorstore has no documents."""
    return len(vectorstore.get(limit=1)["ids"]) == 0
