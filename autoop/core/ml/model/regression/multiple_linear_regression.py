from typing import Dict

from autoop.core.ml.model.regression.base_model import Model

import numpy as np

from pydantic import BaseModel, PrivateAttr


class MultipleLinearRegression(Model, BaseModel):
    """
    Class implementing Multiple Linear Regression model.

    Attributes:
        parameters (Dict[str, np.ndarray]): Dictionary containing the model
        parameters.

    Methods:
        fit(observations: np.ndarray, ground_truth: np.ndarray) -> None:
            Fits the model to the provided observations and ground truth.

        predict(observations: np.ndarray) -> np.ndarray:
            Predicts outcomes based on the provided observations.
    """

    _parameters: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        The normal equation used to calculate the weights is:
        w = (X^T * X)^-1 * X^T * y

        Args:
            observations (np.ndarray): The observations as a 2D array with
                                    samples in rows and features in columns.
            ground_truth (np.ndarray): The ground truth as a 1D array with
                                    the same number of samples as observations.

        Raises:
            ValueError: If the number of samples in observations
                        and ground_truth does not match.
        """

        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                "Number of samples in observations \
                    and ground_truth must match."
            )

        # Add a column of ones to observations for the intercept
        ones = np.ones((observations.shape[0], 1))
        observations_one = np.concatenate((ones, observations), axis=1)

        # Calculate weights using the normal equation
        self._parameters["weights"] = np.linalg.inv(
            observations_one.T @ observations_one
        ) @ (observations_one.T @ ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict outcomes based on the provided observations.

        Args:
            observations (np.ndarray): The observations (features) as a 2D
                            array with samples in rows and features in columns.

        Returns:
            np.ndarray: The predicted outcomes as a 1D array.

        Raises:
            ValueError: If observations are not a 2D array.
        """
        # Validate dimensions
        if observations.ndim != 2:
            raise ValueError("Observations must be a 2D array.")

        # Add a column of ones for the intercept term
        ones = np.ones((observations.shape[0], 1))
        observations_one = np.concatenate((ones, observations), axis=1)

        # Calculate predictions
        return observations_one @ self._parameters["weights"]

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Access the model parameters.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the model parameters.
        """
        return self._parameters
