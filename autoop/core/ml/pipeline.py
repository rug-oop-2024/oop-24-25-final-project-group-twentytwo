from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """Pipeline class for training and evaluating a machine learning model.

    This class manages the training and evaluation of a machine learning model
    by organizing the data preprocessing, model training,
    and evaluation metrics.

    Attributes:
        metrics (List[Metric]): A list of metrics to evaluate the model.
        dataset (Dataset): The dataset to be used for training and evaluation.
        model (Model): The model to be trained.
        input_features (List[Feature]): A list of input features used in the
                                        model.
        target_feature (Feature): The target feature for prediction.
        split (float): The ratio for splitting the dataset into training and
                        testing sets.
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """Initialize the Pipeline.

        Args:
            metrics (List[Metric]): A list of metrics to evaluate the model.
            dataset (Dataset): The dataset to be used for training
                                and evaluation.
            model (Model): The model to be trained.
            input_features (List[Feature]): A list of input features used in
                                            the model.
            target_feature (Feature): The target feature for prediction.
            split (float): The ratio for splitting the dataset into
                                    training and testing sets. Defaults to 0.8.

        Raises:
            ValueError: If the target feature type doesn't
                        match the model type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_feature.type == "categorical"
            and model.type != "classification"  # noqa: W503
        ):
            raise ValueError(
                "Model type must be classification\
                for categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """String representation of the Pipeline.

        Returns:
            str: The string representation of the Pipeline.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Get the model used in the pipeline.

        Returns:
            Model: The model used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Get the artifacts generated during the pipeline execution.

        Returns:
            List[Artifact]: A list of artifacts, including encoders, scalers,
            and model data.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: dict) -> None:
        """Register an artifact in the pipeline.

        Args:
            name (str): The name of the artifact.
            artifact (dict): The artifact data.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocess the features (input and target) in the dataset."""
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """Split the data into training and testing sets."""
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenate feature vectors into a single vector.

        Args:
            vectors (List[np.ndarray]): A list of feature vectors to be
                                        concatenated.

        Returns:
            np.ndarray: The concatenated feature vectors.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model using the training data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluate the model using the test data and compute metrics."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """Execute the pipeline: preprocess, train, and evaluate the model.

        Returns:
            dict: A dictionary containing train and test metrics
                    and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        train_X = self._compact_vectors(self._train_X)
        train_Y = self._train_y
        train_predictions = self._model.predict(train_X)
        train_metrics_result = []

        for metric in self._metrics:
            result = metric.evaluate(train_predictions, train_Y)
            train_metrics_result.append((metric, result))

        self._evaluate()
        test_metrics_results = self._metrics_results
        test_predictions = self._predictions

        return {
            "train_metrics": train_metrics_result,
            "train_predictions": train_predictions,
            "test_metrics": test_metrics_results,
            "test_predictions": test_predictions,
        }
