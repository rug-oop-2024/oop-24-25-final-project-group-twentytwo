"""
Module containing implementations of classification models.

This module includes:
    - RandomForestClassifierModel: A wrapper for scikit-learn's
                                   Random Forest Classifier.
    - DecisionTreeClassifierModel: A wrapper for scikit-learn's
                                   Decision Tree Classifier.
    - KNeighborsClassifierModel: A wrapper for scikit-learn's
                                 K-Nearest Neighbors Classifier.

Each model class provides methods for fitting and predicting,
they are implemented as subclasses of the abstract Model class.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from autoop.core.ml.model.model import Model


# Logistic Regression Classifier
class RandomForestClassifierModel(Model):
    """
    Random Forest Classifier Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self,
        name: str = "RandomForestClassifier",
        type: str = "classification",
    ) -> None:
        """
        Initialize RandomForestClassifierModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "classification".
        """
        super().__init__(name, type)
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest Classifier on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Random Forest Classifier.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)


# Decision Tree Classifier
class DecisionTreeClassifierModel(Model):
    """
    Decision Tree Classifier Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self,
        name: str = "DecisionTreeClassifier",
        type: str = "classification",
    ) -> None:
        """
        Initialize DecisionTreeClassifierModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "classification".
        """
        super().__init__(name, type)
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Decision Tree Classifier on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Decision Tree Classifier.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)


# K-Nearest Neighbors Classifier
class KNeighborsClassifierModel(Model):
    """
    K-Nearest Neighbors Classifier Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self, name: str = "KNeighborsClassifier", type: str = "classification"
    ) -> None:
        """
        Initialize KNeighborsClassifierModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "classification".
        """
        super().__init__(name, type)
        self.model = KNeighborsClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the K-Nearest Neighbors Classifier on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained K-Nearest Neighbors Classifier.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)
