from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    feature_list = []

    data = dataset.read()

    for column_name in data.columns:
        if data[column_name].dtype in ["int64", "float64"]:
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column_name, type=feature_type)
        feature_list.append(feature)

    return feature_list
