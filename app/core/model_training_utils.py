import multiprocessing
from functools import partial
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline

def train_and_evaluate_model(model_name, dataset, input_features, target_feature, metrics, split):
    """
    Train and evaluate a single model using the specified dataset, features, and metrics.
    """
    model = get_model(model_name)
    pipeline = Pipeline(
        metrics=metrics,
        dataset=dataset,
        model=model,
        input_features=input_features,
        target_feature=target_feature,
        split=split
    )
    results = pipeline.execute()
    return {
        "model_name": model_name,
        "train_metrics": results["train_metrics"],
        "test_metrics": results["test_metrics"]
    }

def evaluate_models_parallel(models, dataset, input_features, target_feature, metrics, split):
    """
    Train and evaluate multiple models in parallel.
    """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        partial_function = partial(
            train_and_evaluate_model,
            dataset=dataset,
            input_features=input_features,
            target_feature=target_feature,
            metrics=metrics,
            split=split
        )
        results = pool.map(partial_function, models)
    return results