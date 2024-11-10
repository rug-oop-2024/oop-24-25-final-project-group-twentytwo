"""
Module containing implementations of regression models.

This module includes:
    - LinearRegressionModel: A wrapper for scikit-learn's Linear Regression.
    - DecisionTreeRegressorModel: A wrapper for scikit-learn's
                                  Decision Tree Regressor.
    - KNeighborsRegressorModel: A wrapper for scikit-learn's
                                K-Nearest Neighbors Regressor.
    - MultipleLinearRegressionModel: A Multiple Linear Regression model.

Each model class provides methods for fitting and predicting,
they handle both initialization and model-specific configurations as subclasses
of the abstract Model class.
"""
from autoop.core.ml.model.regression.multiple_linear_regression \
    import MultipleLinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from autoop.core.ml.model.model import Model


# Linear Regression
class LinearRegressionModel(Model):
    """
    Linear Regression Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self, name: str = "LinearRegression", type: str = "regression"
    ) -> None:
        """
        Initialize LinearRegressionModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "regression".
        """
        super().__init__(name, type)
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Linear Regression model on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Linear Regression model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


# Decision Tree Regressor
class DecisionTreeRegressorModel(Model):
    """
    Decision Tree Regressor Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self, name: str = "DecisionTreeRegressor", type: str = "regression"
    ) -> None:
        """
        Initialize DecisionTreeRegressorModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "regression".
        """
        super().__init__(name, type)
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Decision Tree Regressor model on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Decision Tree Regressor model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


# K-Nearest Neighbors Regressor
class KNeighborsRegressorModel(Model):
    """
    K-Nearest Neighbors Regressor Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self, name: str = "KNeighborsRegressor", type: str = "regression"
    ) -> None:
        """
        Initialize KNeighborsRegressorModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "regression".
        """
        super().__init__(name, type)
        self.model = KNeighborsRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the K-Nearest Neighbors Regressor model on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained K-Nearest Neighbors Regressor model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


class MultipleLinearRegressionModel(Model):
    """
    Wrapper for Multiple Linear Regression Model.

    Inherits:
        Model: Base class for machine learning models.
    """

    def __init__(
        self, name: str = "MultipleLinearRegression", type: str = "regression"
    ) -> None:
        """
        Initialize MultipleLinearRegressionModel.

        Args:
            name (str): The name of the model.
            type (str): The type of the model. Defaults to "regression".
        """
        super().__init__(name, type)
        self.model = MultipleLinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Multiple Linear Regression model on training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Multiple Linear Regression model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)
