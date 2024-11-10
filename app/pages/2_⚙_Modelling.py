import streamlit as st
import pandas as pd
import io
import time
import os
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import (
    get_model,
    _CLASSIFICATION_MODELS,
    _REGRESSION_MODELS,
)
from autoop.core.ml.metric import get_metric, METRICS
from autoop.core.ml.pipeline import Pipeline
from app.core.model_training_utils import (
    train_and_evaluate_model,
    evaluate_models_parallel,
)


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Writes helper text for Streamlit application.

    Args:
        text (str): The helper text to display.
    """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a\
    model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

if len(datasets) == 0:
    st.error("Please upload at least one dataset")
    with st.spinner("You are redirected to the Dataset Management page..."):
        time.sleep(2.5)
        st.switch_page("pages/1_📊_Datasets.py")
else:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset:", dataset_names)
    selected_dataset = next(
        dataset
        for dataset in datasets
        if dataset.name == selected_dataset_name
    )

    data = pd.read_csv(io.StringIO(selected_dataset.data.decode()))
    st.write("Preview")
    st.write(data.head())

    if not isinstance(selected_dataset, Dataset):
        selected_dataset = Dataset(
            name=selected_dataset.name,
            asset_path=selected_dataset.asset_path,
            data=selected_dataset.data,
            version=selected_dataset.version,
        )

    features = detect_feature_types(selected_dataset)

    # Separate features into selectable input and target lists
    input_feature_names = [
        feature.name
        for feature in features
        if feature.type == "numerical" or feature.type == "categorical"
    ]
    target_feature_names = [
        feature.name
        for feature in features
        if feature.type == "categorical" or feature.type == "numerical"
    ]

    # Selection of input and target features
    selected_input_features = st.multiselect(
        "Select input features:", input_feature_names
    )
    selected_target_feature = st.selectbox(
        "Select target feature:", target_feature_names
    )

    # Determine task type based on the target feature's type
    target_feature_type = next(
        feature.type
        for feature in features
        if feature.name == selected_target_feature
    )
    task_type = (
        "Classification"
        if target_feature_type == "categorical"
        else "Regression"
    )
    st.write(f"Detected task type: {task_type}")

    # Prompt user to select model based on task type
    st.subheader("Select Model")
    available_models = (
        _CLASSIFICATION_MODELS
        if task_type == "Classification"
        else _REGRESSION_MODELS
    )
    selected_model_name = st.selectbox("Choose a model:", available_models)
    selected_model = get_model(selected_model_name)

    # Prompt user to select dataset split
    st.subheader("Dataset Split")
    train_size = st.slider("Train set size (%):", 10, 90, 80) / 100

    # Metric selection based on task type
    st.subheader("Select Metrics")
    available_metrics = [
        name
        for name, metric in METRICS.items()
        if metric._task_type == task_type
    ]
    selected_metric_names = st.multiselect(
        "Choose metrics:", available_metrics
    )
    selected_metrics = [get_metric(name) for name in selected_metric_names]

    with st.popover(
        "Open Pipeline Summary", use_container_width=True, icon="✨"
    ):
        st.markdown("### 🗂 Selected Dataset")
        st.write(f"**{selected_dataset_name}**")

        st.markdown("### 🔢 Input Features")
        st.write(
            ", ".join(selected_input_features)
            if selected_input_features
            else "None selected"
        )

        st.markdown("### 🎯 Target Feature")
        st.write(f"**{selected_target_feature}**")

        st.markdown("### 🧩 Task Type")
        st.write(f"**{task_type}**")

        st.markdown("### 🤖 Model")
        st.write(f"**{selected_model_name}**")

        st.markdown("### 📊 Train Set Size")
        st.write(f"{train_size * 100:.0f}%")

        st.markdown("### 📏 Metrics")
        st.write(
            ", ".join(selected_metric_names)
            if selected_metric_names
            else "None selected"
        )

    # Initialize features for the pipeline
    input_features = [
        feature
        for feature in features
        if feature.name in selected_input_features
    ]
    target_feature = next(
        feature
        for feature in features
        if feature.name == selected_target_feature
    )

    # Train model and evaluate using Pipeline
    if st.button(
        "Train Selected Model", help="Click to start training", icon="🔥"
    ):
        if not selected_input_features:
            st.error("Please select at least one input feature.")
        elif not selected_target_feature:
            st.error("Please select a target feature.")
        elif not selected_metric_names:
            st.error("Please select at least one metric.")
        else:
            with st.spinner("Training..."):
                results = train_and_evaluate_model(
                    model_name=selected_model_name,
                    dataset=selected_dataset,
                    input_features=input_features,
                    target_feature=target_feature,
                    metrics=selected_metrics,
                    split=train_size,
                )

                # Display results
                st.subheader("📏 Results")

                st.write(f"### Model: {results['model_name']}")
                st.write("**⚙️ Training Metrics**")
                for metric, value in results["train_metrics"]:
                    st.write(f"{metric}: {value:.4f}")

                st.write("**🧪 Testing Metrics**")
                for metric, value in results["test_metrics"]:
                    st.write(f"{metric}: {value:.4f}")

    metric_weights = {}
    num_metrics = len(selected_metric_names)

    # Lock the weight to 1 if there is only one metric
    if num_metrics == 1:
        metric_weights = {selected_metric_names[0]: 1.0}
    else:
        # Only show the Metric Weights section if more than one metric
        # is selected
        if num_metrics > 1:
            st.subheader("Metric Weights")
            metric_weights = {}
            for metric_name in selected_metric_names:
                metric_weights[metric_name] = st.number_input(
                    f"Weight for {metric_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                )

            total_weight = sum(metric_weights.values())
            if total_weight > 0:
                metric_weights = {
                    name: weight / total_weight
                    for name, weight in metric_weights.items()
                }

    if st.button(
        "Evaluate All Models",
        help="Train and evaluate multiple models in parallel",
        icon="⚡",
    ):
        if not selected_input_features:
            st.error("Please select at least one input feature.")
        elif not selected_target_feature:
            st.error("Please select a target feature.")
        elif not selected_metric_names:
            st.error("Please select at least one metric.")
        else:
            with st.spinner("Training models..."):
                results = evaluate_models_parallel(
                    models=available_models,
                    dataset=selected_dataset,
                    input_features=input_features,
                    target_feature=target_feature,
                    metrics=selected_metrics,
                    split=train_size,
                )

                if num_metrics > 1:
                    for result in results:
                        total_score = sum(
                            metric_weights[metric_name] * value
                            for (metric, value) in result["test_metrics"]
                            if metric_name in metric_weights
                        )
                        result["total_score"] = total_score
                else:
                    for result in results:
                        result["total_score"] = 1

                # Select the best model
                best_model = max(results, key=lambda x: x["total_score"])

                st.subheader("🏆 Best Model")
                st.markdown(f"### Model: {best_model['model_name']}")
                st.write("**Training Metrics:**")
                for metric, value in best_model["train_metrics"]:
                    st.write(f"{metric}: {value:.4f}")
                st.write("**Testing Metrics:**")
                for metric, value in best_model["test_metrics"]:
                    st.write(f"{metric}: {value:.4f}")

                with st.popover(
                    "Open Suboptimal Model Metrics",
                    use_container_width=True,
                    icon="🤖",
                ):
                    for result in results:
                        if result != best_model:
                            st.markdown(f"### Model: {result['model_name']}")
                            st.write("**Training Metrics:**")
                            for metric, value in result["train_metrics"]:
                                st.write(f"{metric}: {value:.4f}")
                            st.write("**Testing Metrics:**")
                            for metric, value in result["test_metrics"]:
                                st.write(f"{metric}: {value:.4f}")

    st.subheader("Save Pipeline")
    pipeline_name = st.text_input("Pipeline Name:")
    pipeline_version = st.text_input("Pipeline Version:")

    if st.button("Save Pipeline", help="Save the trained model pipeline:"):
        if not pipeline_name or not pipeline_version:
            st.error("Before saving, provide a name and version")
        else:
            pipeline = Pipeline(
                model=selected_model,
                dataset=selected_dataset,
                input_features=input_features,
                target_feature=target_feature,
                metrics=selected_metrics,
                split=train_size,
            )

        base_path = automl._storage._base_path
        asset_path = os.path.join(base_path, f"{pipeline_name}")
        if pipeline_name:
            serialised_pipeline = pickle.dumps(pipeline)

            artifact = Artifact(
                name=pipeline_name,
                version=pipeline_version,
                asset_path=asset_path,
                tags=[],
                metadata=[],
                data=serialised_pipeline,
                type="pipeline",
            )

            automl = AutoMLSystem.get_instance()
            automl.registry.register(artifact)

            st.success(
                f"Pipeline '{pipeline_name}' v{pipeline_version}\
                has been saved."
            )
