import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression


# 1. Logistic Regression Classifier
class LogisticRegressionModel(Model):
    def __init__(self, name: str = "LogisticRegression"):
        super().__init__(name)
        self.model = LogisticRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 2. Decision Tree Classifier
class DecisionTreeClassifierModel(Model):
    def __init__(self, name: str = "DecisionTreeClassifier"):
        super().__init__(name)
        self.model = DecisionTreeClassifier()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 3. K-Nearest Neighbors Classifier
class KNeighborsClassifierModel(Model):
    def __init__(self, name: str = "KNeighborsClassifier"):
        super().__init__(name)
        self.model = KNeighborsClassifier()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 4. Linear Regression
class LinearRegressionModel(Model):
    def __init__(self, name: str = "LinearRegression"):
        super().__init__(name)
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 5. Decision Tree Regressor
class DecisionTreeRegressorModel(Model):
    def __init__(self, name: str = "DecisionTreeRegressor"):
        super().__init__(name)
        self.model = DecisionTreeRegressor()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 6. K-Nearest Neighbors Regressor
class KNeighborsRegressorModel(Model):
    def __init__(self, name: str = "KNeighborsRegressor"):
        super().__init__(name)
        self.model = KNeighborsRegressor()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
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
    "LogisticRegression",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    models = {
        "LogisticRegression": LogisticRegressionModel,
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
