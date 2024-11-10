from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_features(
    features: List[Feature], dataset: Dataset
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Preprocesses features in the dataset, performing One-Hot Encoding for
    categorical features and Standard Scaling for numerical features.

    Args:
        features (List[Feature]): A list of `Feature` objects that specify
                                  which columns to preprocess.
        dataset (Dataset): The `Dataset` object containing the data to
                           preprocess.

    Returns:
        List[Tuple[str, np.ndarray, dict]]: A list of tuples where each
        tuple contains:
            - The feature name (str),
            - The preprocessed data as a numpy array (np.ndarray),
            - A dictionary containing metadata about the transformation applied
            (dict).
    """
    results = []
    raw = dataset.read()
    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
            ).toarray()
            aritfact = {
                "type": "OneHotEncoder",
                "encoder": encoder.get_params(),
            }
            results.append((feature.name, data, aritfact))
        if feature.type == "numerical":
            scaler = StandardScaler()
            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
            )
            artifact = {
                "type": "StandardScaler",
                "scaler": scaler.get_params(),
            }
            results.append((feature.name, data, artifact))
    # Sort for consistency
    results = list(sorted(results, key=lambda x: x[0]))
    return results
