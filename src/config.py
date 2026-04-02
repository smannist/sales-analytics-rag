class CliMessage:
    """Stores CLI display message text constants."""
    # Query loop related messages
    USER_QUESTION = "[bold cyan]>[/bold cyan] "
    QUERY_HINT = "[bold]Enter a query (Ctrl+C to exit):[/bold]"
    SEARCHING = "[bold green]Searching..."
    NEW_QUERY = "[dim]Enter another query or Ctrl+C to exit.[/dim]"

    # Dataset related messages
    DOWNLOADING = "[yellow]Dataset not found. Downloading from Kaggle...[/yellow]"

    # Vectorstore related messages
    INSERTING = "[bold green]Inserting documents..."
    INSERTED = "[green]Finished inserting all documents.[/green]"
    ALREADY_POPULATED = "[dim]Collection already populated. Skipping.[/dim]"


class DatasetConfig:
    """Stores Kaggle dataset constants."""
    FILENAME = "superstore.csv"
    KAGGLE_HANDLE = "vivek468/superstore-dataset-final"
    KAGGLE_CSV = "Sample - Superstore.csv"
    ENCODING_TYPE = "ISO-8859-1"


class VectorStoreConfig:
    """Configuration for the Chroma vectorstore."""
    PATH = "./chroma_db"
    COLLECTION_NAME = "superstore"
