"""
This package initializes and provides access to machine learning
models for classification and regression tasks.

Modules:
    - Model: Abstract base model class.
    - classification: Contains implementations of classification models.
    - regression: Contains implementations of regression models.

Attributes:
    REGRESSION_MODELS (list): List of available regression model names.
    CLASSIFICATION_MODELS (list): List of available classification model names.

Functions:
    get_model (function): Factory function to retrieve model instance by name.
"""
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import (
    RandomForestClassifierModel,
    DecisionTreeClassifierModel,
    KNeighborsClassifierModel
)
from autoop.core.ml.model.regression import (
    LinearRegressionModel,
    DecisionTreeRegressorModel,
    KNeighborsRegressorModel,
    MultipleLinearRegression
)

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
