from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional
import numpy as np


class Feature(BaseModel):
    """Feature Class.

    Represents a feature in a dataset, with attributes for its name, type,
    optional values, and whether it is the target feature.

    Attributes:
        name (str): The name of the feature.
        type (Literal["categorical", "numerical"]):
            The type of the feature, either categorical or numerical.
        values (Optional[np.ndarray]): An optional array of feature values.
        is_target (bool): Indicates whether this feature is the target feature.
    """

    name: str = Field(..., description="The name of the feature")
    type: Literal["categorical", "numerical"] = Field(
        ..., description="The type of feature"
    )
    values: Optional[np.ndarray] = Field(
        default=None, description="The values of the features"
    )
    is_target: bool = Field(
        default=False,
        description="Indicates whether the feature is the target feature",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        """String representation of the Feature object.

        Returns:
            str: A formatted string with the feature's name, type,
                and target status.
        """
        return f"Feature(name='{self.name}', \
            type='{self.type}', \
            is_target={self.is_target})"
