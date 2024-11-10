"""
This module provides access to machine learning
_models for classification and regression tasks.

Modules:
    - Model: Abstract base model class.
    - classification: Contains implementations of classification _models.
    - regression: Contains implementations of regression _models.

Attributes:
    _REGRESSION_MODELS (list): List of available regression model names.
    _CLASSIFICATION_MODELS (list): List of available classification model
                                   names.

Functions:
    get_model (function): Factory function to retrieve model instance by name.
"""

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import (
    RandomForestClassifierModel,
    DecisionTreeClassifierModel,
    KNeighborsClassifierModel,
)
from autoop.core.ml.model.regression import (
    LinearRegressionModel,
    DecisionTreeRegressorModel,
    KNeighborsRegressorModel,
    MultipleLinearRegression,
)

_REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "LinearRegression",
    "DecisionTreeRegressor",
    "KNeighborsRegressor",
]

_CLASSIFICATION_MODELS = [
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
    _models = {
        "RandomForestClassifier": RandomForestClassifierModel,
        "DecisionTreeClassifier": DecisionTreeClassifierModel,
        "KNeighborsClassifier": KNeighborsClassifierModel,
        "LinearRegression": LinearRegressionModel,
        "DecisionTreeRegressor": DecisionTreeRegressorModel,
        "KNeighborsRegressor": KNeighborsRegressorModel,
        "MultipleLinearRegression": MultipleLinearRegression,
    }

    if model_name in _models:
        return _models[model_name]()
    else:
        raise ValueError(f"Model {model_name} not recognized.")
