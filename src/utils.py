from pathlib import Path

from langchain_core.documents import Document


def load_file(path: str) -> str:
    """Loads a text file.

    Args:
        path: Path to the text file, absolute or relative to this script.

    Returns:
        The contents of the file.
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent / path
    return file_path.read_text(encoding="utf-8")


def rank_by_total_sales(docs: list[Document]) -> list[Document]:
    """Sort documents by Total_Sales descending.

    Args:
        docs: The documents to rank.

    Returns:
        Documents sorted by Total_Sales descending, or the original list
        if any document is missing the field.
    """
    if not all("Total_Sales" in d.metadata for d in docs):
        return docs
    return sorted(
        docs,
        key=lambda d: d.metadata["Total_Sales"],
        reverse=True
    )
