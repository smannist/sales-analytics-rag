from pathlib import Path

import kagglehub
import pandas as pd

from config import DatasetConfig


def download_dataset(
    filename: str = DatasetConfig.FILENAME,
    kaggle_handle: str = DatasetConfig.KAGGLE_HANDLE,
    kaggle_csv: str = DatasetConfig.KAGGLE_CSV,
    encoding_type: str = DatasetConfig.ENCODING_TYPE
) -> None:
    """Download the dataset from Kaggle and save it locally.

    Args:
        filename: The name of the file to save the dataset as.
        kaggle_handle: The Kaggle dataset handle.
        kaggle_csv: The CSV filename within the Kaggle dataset.
        encoding_type: The CSV encoding type
    """
    base_path = kagglehub.dataset_download(kaggle_handle)
    dataset_path = Path(base_path) / kaggle_csv
    df = pd.read_csv(dataset_path, encoding=encoding_type)
    df.to_csv(filename, index=False)


def load_dataset(
        filename: str = DatasetConfig.FILENAME,
        encoding_type: str = DatasetConfig.ENCODING_TYPE
) -> pd.DataFrame:
    """Load the dataset from a local CSV file, if not present, download first.

    Args:
        filename: The name of the CSV file to load.
        encoding_type: The CSV encoding type

    Returns:
        The loaded dataset as a pandas DataFrame.
    """
    try:
        return pd.read_csv(filename, encoding=encoding_type)
    except FileNotFoundError:
        download_dataset(filename)
        return pd.read_csv(filename, encoding=encoding_type)
