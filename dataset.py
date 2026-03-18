import pandas as pd
import kagglehub

from pathlib import Path


def download_and_save_kaggle_dataset(
    dataset_location: str,
    dataset_name: str,
    output_path: str,
    encoding: str = "utf-8"
) -> None:
    """
    Downloads a dataset from Kaggle to cache using Kagglehub and saves it as a local CSV file.

    Args:
        dataset_location: The location of the dataset on Kaggle.
        dataset_name: The name of the dataset file.
        output_path: The local path to save the CSV file to.
        encoding: The encoding of the dataset (optional, default is utf-8).
    """
    base_path = kagglehub.dataset_download(dataset_location)
    df = pd.read_csv(Path(base_path) / dataset_name, encoding=encoding)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    download_and_save_kaggle_dataset(
        dataset_location="vivek468/superstore-dataset-final",
        dataset_name="Sample - Superstore.csv",
        output_path="superstore.csv",
        encoding="ISO-8859-1",
    )
