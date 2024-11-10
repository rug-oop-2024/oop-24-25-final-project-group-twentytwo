from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects the types of features in the given dataset.
    Assumes that the dataset only contains categorical and numerical features,
    and that there are no NaN values.

    Args:
        dataset (Dataset): The dataset object that contains the data.

    Returns:
        List[Feature]: A list of `Feature` objects with their names
        and detected types (either 'numerical' or 'categorical')
        for each column in the dataset.
    """
    _feature_list = []

    data = dataset.read()

    for column_name in data.columns:
        if data[column_name].dtype in ["int64", "float64"]:
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column_name, type=feature_type)
        _feature_list.append(feature)

    return _feature_list
