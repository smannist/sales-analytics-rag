class CliMessage:
    """Stores CLI display message text constants."""
    # General
    WELCOME = "Welcome to Superstore Analytics!"

    # Query loop related constants
    INPUT_PROMPT = "> "
    QUERY_HINT = "Enter a query (Ctrl+C to exit):"
    SEARCHING = "Searching..."
    FOLLOWING_UP = "Following up..."
    NEW_QUERY = "Enter another query or Ctrl+C to exit."

    # Dataset related constants
    DATASET_EXISTS = "Dataset already exists. Skipping download."
    DATASET_DOWNLOADING = "Dataset not found. Downloading from Kaggle..."
    DATASET_DOWNLOADED = "Dataset downloaded successfully."

    # Vectorstore related constants
    INSERTING = "Inserting documents..."
    INSERTED = "Finished inserting all documents."
    ALREADY_POPULATED = "Collection already populated. Skipping."


class DatasetConfig:
    """Stores Kaggle dataset constants."""
    FILENAME = "superstore.csv"
    KAGGLE_HANDLE = "vivek468/superstore-dataset-final"
    KAGGLE_CSV = "Sample - Superstore.csv"
    ENCODING_TYPE = "ISO-8859-1"


class VectorDBConfig:
    """Stores vectorDB constants."""
    PATH = "./chroma_db"
    COLLECTION_NAME = "superstore"
    INSERTION_BATCH_SIZE = 500


class OpenAIConfig:
    """Stores OpenAI constants."""
    MODEL = "gpt-4o-mini"


class ChatHistoryConfig:
    """Stores chat history constants."""
    FILE_PATH = "./chat_history.json"
    MAX_HISTORY_TOKENS = 6000
