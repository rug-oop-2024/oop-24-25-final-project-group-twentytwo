from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import base64
from typing import Optional, Dict, List

class Artifact(BaseModel, ABC):
    asset_path: str = Field(..., description="Path to asset")
    version: str = Field(..., description="Version of artefact")
    data: bytes = Field(..., description="Binary data of the asset")
    metadata: Optional[Dict[str,str]] = Field(default_factory=dict, description="additional data")
    type: str = Field(..., description="Type of the artifact")
    tags: List[str] = Field(default_factory=list, description="Tags for categorising the artifact")

    @property
    def id(self) -> str:
        """Generates an ID based on the asset path and version"""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    @abstractmethod
    def save(self, data: bytes) -> bytes:
        """Abstract mehtod to save data into the artifact."""
        pass

    @abstractmethod
    def read(self) -> bytes:
        """Abstract mehtod to read data into the artifact."""
        pass