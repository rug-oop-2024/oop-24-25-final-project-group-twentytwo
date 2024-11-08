from abc import ABC, abstractmethod
import numpy as np
from typing import Literal


class Metric(ABC):
    """Base class for all metrics."""

    task_type: Literal["Regression", "Classification"]

    @abstractmethod
    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the metric with ground truth and predicted values."""
        pass

    def evaluate(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return self.__call__(gt,pred)
    
    def __str__(self):
        return self.__class__.__name__


# Regression Metrics (adapted from: https://machinelearningmastery.com/regression-metrics-for-machine-learning/)


class MeanSquaredError(Metric):
    """Implementation of Mean Squared Error metric."""

    task_type = "Regression"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.mean((gt - pred) ** 2)



class RootMeanSquaredError(Metric):
    """Implementation of Root Squared Error metric."""

    task_type = "Regression"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.sqrt(np.mean((gt - pred) ** 2))


class MeanAbsoluteError(Metric):
    """Implementation of Mean Absolute Error metric."""

    task_type = "Regression"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(np.abs(gt - pred))


# Classification Metrics (adapted from: https://www.appsilon.com/post/machine-learning-evaluation-metrics-classification)


class Accuracy(Metric):
    """Implementation of Accuracy metric."""

    task_type = "Classification"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(gt == pred)


class Recall(Metric):
    """Implementation of Recall metric."""

    task_type = "Classification"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        true_positive = np.sum((pred == 1) & (gt == 1))
        false_negative = np.sum((pred == 0) & (gt == 1))
        return (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )


class Precision(Metric):
    """Implementation of Precision metric."""

    task_type = "Classification"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        true_positive = np.sum((pred == 1) & (gt == 1))
        false_positive = np.sum((pred == 1) & (gt == 0))
        return (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )


METRICS = {
    "Mean Squared Error": MeanSquaredError,
    "Root Mean Squared Error": RootMeanSquaredError,
    "Mean Absolute Error": MeanAbsoluteError,
    "Accuracy": Accuracy,
    "Recall": Recall,
    "Precision": Precision,
}


def get_metric(name: str) -> Metric:
    """Factory function to return a metric instance by str name."""
    try:
        return METRICS[name]()
    except KeyError:
        raise ValueError(f"Not supported metric: {name}")

