from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Dataset Class.

    Inherits from `Artifact` to provide functionality for handling datasets.
    Adds methods for creating a `Dataset` instance from a pandas DataFrame
    and reading/saving dataset data in CSV format.

    Methods:
        from_dataframe(data: pd.DataFrame, name: str,
            asset_path: str, version: str) -> Dataset:
            Creates a `Dataset` instance from a pandas DataFrame.

        read() -> pd.DataFrame:
            Reads the dataset's binary data as a pandas DataFrame.

        save(data: pd.DataFrame) -> bytes:
            Saves a pandas DataFrame as binary data in CSV format.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialises a Dataset instance.

        Parameters:
            *args: Positional arguments for the parent class initializer.
            **kwargs: Keyword arguments for the parent class initializer.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """Creates a Dataset instance from a pandas DataFrame.

        Parameters:
            data (pd.DataFrame): The DataFrame to convert into a Dataset.
            name (str): Name of the dataset.
            asset_path (str): Path to save the dataset asset.
            version (str): Version of the dataset.
                Defaults to "1.0.0".

        Returns:
            Dataset: A Dataset instance containing the DataFrame as CSV data.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Reads the dataset's binary data and converts it
            to a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Saves a pandas DataFrame as binary data in CSV format.

        Parameters:
            data (pd.DataFrame): The DataFrame to save.

        Returns:
            bytes: The binary CSV data saved in the parent Artifact class.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
