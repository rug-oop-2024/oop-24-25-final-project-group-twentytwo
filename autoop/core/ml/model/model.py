from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Any



class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self._parameters = {}
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, filepath: str) -> None:
        artifact = Artifact(model=deepcopy(self), filepath=filepath)
        artifact.save()
        print(f"Model saved at {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Model":
        artifact = Artifact.load(filepath)
        print(f"Model loaded from {filepath}")
        return artifact.model

    def set_parameters(self, **params: Any) -> None:
        self._parameters.update

    def get_parameters(self) -> dict:
        return deepcopy(self._parameters)
