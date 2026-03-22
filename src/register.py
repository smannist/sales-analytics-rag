from typing import Callable

import pandas as pd

from document import Document

DOCUMENT_REGISTRY: list[Callable[[pd.DataFrame], Document]] = []


def register(
    factory: Callable[[pd.DataFrame], Document],
) -> Callable[[pd.DataFrame], Document]:
    """Registers a document factory function.

    Args:
        factory: A factory function that takes a DataFrame and returns a Document.

    Returns:
        The factory function
    """
    DOCUMENT_REGISTRY.append(factory)
    return factory
