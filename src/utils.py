from pathlib import Path


def load_file(file: str, path: str) -> str:
    """Loads a text file.

    Args:
        file: The boilerplate caller file.
        path: Relative path to the text file.

    Returns:
        The contents of the file.
    """
    return (Path(file).parent / path).read_text(encoding="utf-8")
