from collections.abc import Callable

import pandas as pd
from langchain_core.documents import Document

DOCUMENT_REGISTRY: dict[str, Callable[[pd.DataFrame], list[Document]]] = {}


def register(
    factory: Callable[[pd.DataFrame], list[Document]],
) -> Callable[[pd.DataFrame], list[Document]]:
    """Registers a document factory function.

    Args:
        factory: A factory function that takes a DataFrame
                 and returns a list of Document(s).

    Returns:
        The factory function
    """
    DOCUMENT_REGISTRY[factory.__name__] = factory
    return factory
