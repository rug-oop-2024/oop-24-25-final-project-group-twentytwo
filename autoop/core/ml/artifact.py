from pydantic import BaseModel, Field
import base64
import re
from typing import Optional, Dict, List

class Artifact(BaseModel):
    asset_path: str = Field(..., description="Path to asset")
    name: str = Field(..., description="Name of the artifact")
    version: str = Field(..., description="Version of artifact")
    data: Optional[bytes] = Field(..., description="Binary data of the asset")
    metadata: Optional[Dict[str,str]] = Field(default_factory=dict, description="additional data")
    type: str = Field(..., description="Type of the artifact")
    tags: List[str] = Field(default_factory=list, description="Tags for categorising the artifact")

    @property
    def id(self) -> str:
        """Generates an ID based on the asset path and version"""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode().rstrip("=")
        safe_version = re.sub(r'[:;,.=]', '_', self.version)
        return f"{encoded_path}:{safe_version}"

    def save(self, data: bytes) -> bytes:
        """Abstract method to save data into the artifact."""
        self.data = data

    def read(self) -> bytes:
        """Abstract method to read data into the artifact."""
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data