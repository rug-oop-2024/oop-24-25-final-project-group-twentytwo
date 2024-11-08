import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression


# 1. Logistic Regression Classifier
class RandomForestClassifierModel(Model):
    def __init__(self, name: str = "RandomForestClassifier", type = "classification"):
        super().__init__(name, type)
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 2. Decision Tree Classifier
class DecisionTreeClassifierModel(Model):
    def __init__(self, name: str = "DecisionTreeClassifier", type = "classification"):
        super().__init__(name, type)
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 3. K-Nearest Neighbors Classifier
class KNeighborsClassifierModel(Model):
    def __init__(self, name: str = "KNeighborsClassifier", type = "classification"):
        super().__init__(name, type)
        self.model = KNeighborsClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 4. Linear Regression
class LinearRegressionModel(Model):
    def __init__(self, name: str = "LinearRegression", type="regression"):
        super().__init__(name, type)
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 5. Decision Tree Regressor
class DecisionTreeRegressorModel(Model):
    def __init__(self, name: str = "DecisionTreeRegressor", type = "regression"):
        super().__init__(name, type)
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 6. K-Nearest Neighbors Regressor
class KNeighborsRegressorModel(Model):
    def __init__(self, name: str = "KNeighborsRegressor", type = "regression"):
        super().__init__(name, type)
        self.model = KNeighborsRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
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
    """Factory function to get a model by name."""
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
