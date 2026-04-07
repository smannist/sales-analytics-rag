from collections.abc import Iterator

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import VectorDBConfig


# could use sentence-transformers package, but its gigantic
# default seems fine, so no need to use openai either in this simple app
class ChromaOnnxEmbeddings(Embeddings):
    """LC wrapper around Chroma's default all-MiniLM-L6-v2 embeddings."""

    def __init__(self) -> None:
        """The initializer."""
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vector representations.

        Args:
            texts: A list of strings to embed.

        Returns:
            The embedding vectors.
        """
        return [arr.tolist() for arr in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a vector representation.

        Args:
            text: The query string to embed.

        Returns:
            A single embedding vector.
        """
        return self._fn([text])[0].tolist()


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
        embedding_function=ChromaOnnxEmbeddings(),
    )


def is_vectorstore_empty(vectorstore: Chroma) -> bool:
    """Return True if the vectorstore has no documents."""
    return len(vectorstore.get(limit=1)["ids"]) == 0
