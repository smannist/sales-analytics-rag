from collections.abc import Callable

import pandas as pd
from langchain_core.documents import Document


type DocumentFactory = Callable[[pd.DataFrame], list[Document]]

DOCUMENT_FACTORY_REGISTRY: list[DocumentFactory] = []


def document_factory(
    factory: DocumentFactory,
) -> DocumentFactory:
    """Registers a document factory function.

    Args:
        factory: A factory function that takes a DataFrame
                 and returns a list of Document(s).

    Returns:
        The factory function
    """
    DOCUMENT_FACTORY_REGISTRY.append(factory)
    return factory


def load_document_factories() -> list[DocumentFactory]:
    """Imports factories, registers them, returns the registry.

    Returns:
        The populated document factory registry.
    """
    import factories  # noqa: F401, PLC0415 -- local import breaks the registry<->factories cycle; imported for decorator side effects, maybe it just better to have gigantic list and manually insert
    return DOCUMENT_FACTORY_REGISTRY
