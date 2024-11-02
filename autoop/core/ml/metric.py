from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the metric with ground truth and predicted values."""
        pass


# Regression Metrics (adapted from: https://machinelearningmastery.com/regression-metrics-for-machine-learning/)


class MeanSquaredError(Metric):
    """Implementation of Mean Squared Error metric."""

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.mean((gt - pred) ** 2)


class RootMeanSquaredError(Metric):
    """Implementation of Root Squared Error metric."""

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.sqrt(np.mean((gt - pred) ** 2))


class MeanAbsoluteError(Metric):
    """Implementation of Mean Absolute Error metric."""

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(np.abs(gt - pred))


# Classification Metrics (adapted from: https://www.appsilon.com/post/machine-learning-evaluation-metrics-classification)


class Accuracy(Metric):
    """Implementation of Accuracy metric."""

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(gt == pred)


class Recall(Metric):
    """Implementation of Recall metric."""

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

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        true_positive = np.sum((pred == 1) & (gt == 1))
        false_positive = np.sum((pred == 1) & (gt == 0))
        return (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )


METRICS = {
    "mean_squared_error": MeanSquaredError,
    "root_mean_squared_error": RootMeanSquaredError,
    "mean_absolute_error": MeanAbsoluteError,
    "accuracy": Accuracy,
    "recall": Recall,
    "precision": Precision,
}


def get_metric(name: str) -> Metric:
    """Factory function to return a metric instance by str name."""
    try:
        return METRICS[name]()
    except KeyError:
        raise ValueError(f"Not supported metric: {name}")
