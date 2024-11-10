from abc import ABC, abstractmethod
import numpy as np
from typing import Literal


class Metric(ABC):
    """Base class for all metrics.

    This class serves as a base for all metric implementations. It defines
    the common interface for computing metrics based on ground truth and
    predicted values.

    Attributes:
        _task_type (Literal["Regression", "Classification"]): Type of task
            the metric is used for, either "Regression" or "Classification".
    """

    _task_type: Literal["Regression", "Classification"]

    @property
    def task_type(self) -> str:
        """Get the task type for the metric.

        Returns:
            str: task type.
        """
        return self._task_type

    @abstractmethod
    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the metric with ground truth and predicted values.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The computed metric value.
        """
        pass

    def evaluate(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Evaluate the metric using ground truth and predictions.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The evaluated metric value.
        """
        return self.__call__(gt, pred)

    def __str__(self) -> str:
        """String representation of the Metric class.

        Returns:
            str: The name of the metric class.
        """
        return self.__class__.__name__


# Regression Metrics: (adapted from:
# https://machinelearningmastery.com/regression-metrics-for-machine-learning/)


class MeanSquaredError(Metric):
    """Implementation of Mean Squared Error metric.

    This metric computes the mean squared error between ground truth
    and predicted values.

    Task type: "Regression"
    """

    _task_type = "Regression"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the Mean Squared Error.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The mean squared error value.
        """
        return np.mean((gt - pred) ** 2)


class RootMeanSquaredError(Metric):
    """Implementation of Root Mean Squared Error metric.

    This metric computes the square root of the mean squared error between
    ground truth and predicted values.

    Task type: "Regression"
    """

    _task_type = "Regression"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the Root Mean Squared Error.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The root mean squared error value.
        """
        return np.sqrt(np.mean((gt - pred) ** 2))


class MeanAbsoluteError(Metric):
    """Implementation of Mean Absolute Error metric.

    This metric computes the mean absolute error between ground truth
    and predicted values.

    Task type: "Regression"
    """

    _task_type = "Regression"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the Mean Absolute Error.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The mean absolute error value.
        """
        return np.mean(np.abs(gt - pred))


# Classification Metrics (adapted from:
# https://www.appsilon.com/post/machine-learning-evaluation-metrics-classification)


class Accuracy(Metric):
    """Implementation of Accuracy metric.

    This metric computes the accuracy between ground truth
    and predicted values.

    Task type: "Classification"
    """

    _task_type = "Classification"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the Accuracy.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The accuracy value.
        """
        return np.mean(gt == pred)


class Recall(Metric):
    """Implementation of Recall metric.

    This metric computes the recall between ground truth and predicted values.

    Task type: "Classification"
    """

    _task_type = "Classification"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the Recall.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The recall value.
        """
        true_positive = np.sum((pred == 1) & (gt == 1))
        false_negative = np.sum((pred == 0) & (gt == 1))
        return (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )


class Precision(Metric):
    """Implementation of Precision metric.

    This metric computes the precision between ground
    truth and predicted values.

    Task type: "Classification"
    """

    _task_type = "Classification"

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute the Precision.

        Parameters:
            gt (np.ndarray): The ground truth values.
            pred (np.ndarray): The predicted values.

        Returns:
            float: The precision value.
        """
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
    """Factory function to return a metric instance by its string name.

    Parameters:
        name (str): The name of the metric.

    Returns:
        Metric: The corresponding metric instance.

    Raises:
        ValueError: If the metric name is not supported.
    """
    try:
        return METRICS[name]()
    except KeyError:
        raise ValueError(f"Not supported metric: {name}")
