from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Any


class Model(ABC):
    """
    Abstract base class for machine learning models. Provides a structure
    for implementing custom models for classification or regression tasks.
    """

    def __init__(
        self, name: str, type: Literal["classification", "regression"]
    ) -> None:
        """
        Initialize the Model with a name and type.

        Args:
            name (str): The name of the model.
            type (Literal["classification", "regression"]): The type of task
                the model is designed for.
        """
        self.name = name
        self._parameters = {}
        self.is_trained = False
        self.type = type

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target array of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predictions of shape (n_samples,).
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath (str): Path to save the model artifact.
        """
        artifact = Artifact(model=deepcopy(self), filepath=filepath)
        artifact.save()
        print(f"Model saved at {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Model":
        """
        Load a model from a file.

        Args:
            filepath (str): Path to the saved model artifact.

        Returns:
            Model: The loaded model instance.
        """
        artifact = Artifact.load(filepath)
        print(f"Model loaded from {filepath}")
        return artifact.model

    def set_parameters(self, **params: Any) -> None:  # Noqa: ANN401
        """
        Set hyperparameters or configurations for the model.

        Args:
            **params (Any): Key-value pairs of parameters to set.
        """
        self._parameters.update

    def get_parameters(self) -> dict:
        """
        Get the current parameters of the model.

        Returns:
            dict: A dictionary of the model's parameters.
        """
        return deepcopy(self._parameters)
