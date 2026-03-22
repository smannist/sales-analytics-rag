from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Holds the document name, data and metadata for a single vectorDB document."""
    name: str
    data: list[str]
    metadata: list[dict[str, str | float | int]] | None = None
