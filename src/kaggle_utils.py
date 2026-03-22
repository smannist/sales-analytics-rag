from pathlib import Path

import kagglehub
import pandas as pd


def download_and_save_csv(
    dataset_location: str,
    dataset_name: str,
    output_path: str,
    encoding: str = "utf-8"
) -> None:
    """Downloads a dataset from Kaggle and saves it as a local CSV file.

    Args:
        dataset_location: The location of the dataset on Kaggle.
        dataset_name: The name of the dataset file.
        output_path: The local path to save the CSV file to.
        encoding: The encoding of the dataset (optional, default is utf-8).

    Raises:
        RuntimeError: If the dataset download or save fails.
    """
    try:
        base_path = kagglehub.dataset_download(dataset_location)
        df = pd.read_csv(Path(base_path) / dataset_name, encoding=encoding)
        df.to_csv(output_path, index=False)
        print(f"Finished. Dataset saved to {output_path}.")
    except Exception as e:
        msg = f"Error downloading dataset: {e}"
        raise RuntimeError(msg) from e
