from pydantic import BaseModel, Field
from typing import Literal, Optional
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    name: str = Field(..., description="The name of the feature")
    feature_type: Literal['categorical', 'numerical'] = Field(..., description="The type of feature")
    values: Optional[np.ndarray] = Field(default=None, description="The values of the features")
    is_target: bool = Field(default=False, description="Indicates whether the feature is the target feature")

    def __str__(self):
        return f"Feature(name='{self.name}', type='{self.feature_type}', is_target={self.is_target})"
