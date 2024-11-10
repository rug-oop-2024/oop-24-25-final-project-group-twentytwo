import numpy as np  # noqa: D104
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression


# 1. Logistic Regression Classifier
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


# 2. Decision Tree Classifier
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


# 3. K-Nearest Neighbors Classifier
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


# 4. Linear Regression
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


# 5. Decision Tree Regressor
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


# 6. K-Nearest Neighbors Regressor
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


REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "LinearRegression",
    "DecisionTreeRegressor",
    "KNeighborsRegressor",
]

CLASSIFICATION_MODELS = [
    "RandomForestClassifier",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
]


def get_model(model_name: str) -> Model:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        Model: The corresponding model instance.

    Raises:
        ValueError: If the specified model name is not recognized.
    """
    models = {
        "RandomForestClassifier": RandomForestClassifierModel,
        "DecisionTreeClassifier": DecisionTreeClassifierModel,
        "KNeighborsClassifier": KNeighborsClassifierModel,
        "LinearRegression": LinearRegressionModel,
        "DecisionTreeRegressor": DecisionTreeRegressorModel,
        "KNeighborsRegressor": KNeighborsRegressorModel,
        "MultipleLinearRegression": MultipleLinearRegression,
    }

    if model_name in models:
        return models[model_name]()
    else:
        raise ValueError(f"Model {model_name} not recognized.")
