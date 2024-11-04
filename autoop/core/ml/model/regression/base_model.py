from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Model(ABC):
    """
    Abstract base class for models.

    Attributes:
        parameters (dict): A dictionary to store model parameters.

    Methods:
        fit(observations: np.ndarray, ground_truth: np.ndarray) -> None:
            Abstract method that fits
            the model observations and ground truth data.

        predict(observations: np.ndarray) -> np.ndarray:
            Abstract that predicts
            outcomes based on the fitted
            model using the observations.
    """

    _parameters: Dict[str, np.ndarray]

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to provided observation and ground truth data.

        Args:
            observations (np.ndarray): The input data to fit the model.
            ground_truth (np.ndarray): The actual values to train the model
            against.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict outcomes based on the provided observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: The predicted outcomes.
        """
        pass
