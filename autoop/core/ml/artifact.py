import base64
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """Artifact Class.

    The `Artifact` class represents a digital artifact with attributes
        to store its metadata, type, tags, and binary data.
    It provides functionality for managing artifact data,
        including reading, saving, and generating unique identifiers.

    Attributes:
        asset_path (str): Path to the asset.
        name (str): Name of the artifact.
        version (str): Version of the artifact.
        data (Optional[bytes]): Binary data of the asset. Defaults to None.
        metadata (Optional[Dict[str, str]]): Additional metadata
            for the artifact. Defaults to an empty dictionary.
        type (str): The type of the artifact (e.g., "image", "document").
        tags (List[str]): Tags used to categorize the artifact.
            Defaults to an empty list.

    Properties:
        id (str): A unique identifier for the artifact,
            generated using a base64-encoded asset path and version string.

    Methods:
        save(data: bytes) -> bytes:
            Stores binary data into the artifact.

        read() -> bytes:
            Retrieves the binary data of the artifact.
            Raises:
                ValueError: If no data is available to read.
    """

    asset_path: str = Field(..., description="Path to asset")
    name: str = Field(..., description="Name of the artifact")
    version: str = Field(..., description="Version of artifact")
    data: Optional[bytes] = Field(..., description="Binary data of the asset")
    metadata: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="additional data"
    )
    type: str = Field(..., description="Type of the artifact")
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorising the artifact"
    )
    data: bytes

    @property
    def id(self) -> str:
        """Generates an ID based on the asset path and version.

        Returns:
            str: identifier of the artifact.

        """
        encoded_path = (
            base64.b64encode(self.asset_path.encode()).decode().rstrip("=")
        )
        safe_version = re.sub(r"[:;,.=]", "_", self.version)
        return f"{encoded_path}{safe_version}"

    def save(self, data: bytes) -> bytes:
        """Method to save data into the artifact.

        Parameters:
            data (bytes): The binary data to save.
        """
        self.data = data

    def read(self) -> bytes:
        """Retrieves the binary data of the artifact.

        Returns:
            bytes: The binary data of the artifact.

        Raises:
            ValueError: If no data is available to read.
        """
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data
