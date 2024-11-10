import multiprocessing
from functools import partial
from typing import List, Dict, Any
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline


def train_and_evaluate_model(
    model_name: str,
    dataset: Any,  # noqa: ANN401
    input_features: List[str],
    target_feature: str,
    metrics: List[str],
    split: Dict[str, float]
) -> Dict[str, Any]:
    """
    Train and evaluate a single model using the specified dataset, features,
    and metrics.

    Args:
        model_name (str): The name of the model to train and evaluate.
        dataset (Any): The dataset to use for training and testing.
        input_features (List[str]): The list of input feature names.
        target_feature (str): The name of the target feature.
        metrics (List[str]): The list of metrics to evaluate the model on.
        split (Dict[str, float]): A dictionary specifying the train/test split.

    Returns:
        Dict[str, Any]: A dictionary containing the model name,
        training metrics, and test metrics.
    """
    model = get_model(model_name)
    pipeline = Pipeline(
        metrics=metrics,
        dataset=dataset,
        model=model,
        input_features=input_features,
        target_feature=target_feature,
        split=split,
    )
    results = pipeline.execute()
    return {
        "model_name": model_name,
        "train_metrics": results["train_metrics"],
        "test_metrics": results["test_metrics"],
    }


def evaluate_models_parallel(
    models: List[str],
    dataset: Any,  # noqa: ANN401
    input_features: List[str],
    target_feature: str,
    metrics: List[str],
    split: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Train and evaluate multiple models in parallel.

    Args:
        models (List[str]): The list of model names to train and evaluate.
        dataset (Any): The dataset to use for training and testing.
        input_features (List[str]): The list of input feature names.
        target_feature (str): The name of the target feature.
        metrics (List[str]): The list of metrics to evaluate the models on.
        split (Dict[str, float]): A dictionary specifying the train/test split.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing model names,
        training metrics, and test metrics for each model.
    """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        partial_function = partial(
            train_and_evaluate_model,
            dataset=dataset,
            input_features=input_features,
            target_feature=target_feature,
            metrics=metrics,
            split=split,
        )
        results = pool.map(partial_function, models)
    return results
