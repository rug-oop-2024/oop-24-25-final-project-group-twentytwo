from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
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
    _values: Optional[np.ndarray] = PrivateAttr(default=None)
    _is_target: bool = PrivateAttr(
        default=False,
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def values(self) -> Optional[np.ndarray]:
        """Getter for the feature's values.

        Returns:
            Optional[np.ndarray]: The values of the feature,
            or None if not set.
        """
        return self._values

    @values.setter
    def values(self, value: Optional[np.ndarray]) -> None:
        """Setter for the feature's values with validation.

        Args:
            value (Optional[np.ndarray]): The new values to
                                         assign to the feature.

        Raises:
            ValueError: If the values are not of the expected type.
        """
        if value is not None and not isinstance(value, np.ndarray):
            raise ValueError("Values must be a numpy.ndarray.")
        self._values = value

    @property
    def is_target(self) -> bool:
        """Getter for the is_target status.

        Returns:
            bool: Whether the feature is the target feature.
        """
        return self._is_target

    @is_target.setter
    def is_target(self, value: bool) -> None:
        """Setter for the is_target flag.

        Args:
            value (bool): Whether the feature is the target feature.

        Raises:
            ValueError: If the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("is_target must be a boolean.")
        self._is_target = value

    def __str__(self) -> str:
        """String representation of the Feature object.

        Returns:
            str: A formatted string with the feature's name, type,
                and target status.
        """
        return f"Feature(name='{self.name}', \
            type='{self.type}', \
            is_target={self.is_target})"
