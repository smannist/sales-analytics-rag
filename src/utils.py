from pathlib import Path


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
