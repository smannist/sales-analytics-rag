from dataclasses import dataclass
from chromadb.types import Metadata


@dataclass(frozen=True)
class Document:
    """Holds the document name, data and metadata for a single vectorDB document."""
    name: str
    data: list[str]
    metadata: list[Metadata] | None = None
