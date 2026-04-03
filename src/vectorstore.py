from collections.abc import Iterator

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
    batch_size = 500
    total = len(documents)
    for i in range(0, total, batch_size):
        vectorstore.add_documents(documents[i : i + batch_size])
        yield min(i + batch_size, total), total


def get_vectorstore() -> Chroma:
    """Returns a Chroma vectorstore.

    Returns:
        The Chroma vectorstore instance.
    """
    return Chroma(
        client=chromadb.PersistentClient(path=VectorDBConfig.PATH),
        collection_name=VectorDBConfig.COLLECTION_NAME
    )


def is_vectorstore_empty(vectorstore: Chroma) -> bool:
    """Return True if the vectorstore has no documents."""
    return vectorstore._collection.count() == 0  # noqa: SLF001
