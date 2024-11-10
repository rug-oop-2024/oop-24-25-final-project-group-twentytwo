import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

st.title("Dataset Management")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    if st.toggle("Preview"):
        st.write("Preview of the uploaded dataset:")
        st.write(df.head())

    dataset_name = st.text_input("Enter a name for the dataset")
    version_input = st.text_input("Enter a version")

    base_path = automl._storage._base_path
    asset_path = os.path.join(base_path, f"{dataset_name}.csv")

    if st.button("Save Dataset"):
        if not dataset_name:
            st.error("Please enter a name for the dataset.")
        elif not version_input:
            st.error("Please enter a version for the dataset.")
        else:
            dataset = Dataset.from_dataframe(
                df,
                name=dataset_name,
                asset_path=asset_path,
                version=version_input,
            )
            automl.registry.register(dataset)
            st.write(f"Dataset saved to: {asset_path}")
            st.success(
                f"Dataset '{dataset_name}' v{version_input}\
                saved successfully!"
            )

# List existing datasets with delete option
st.subheader("Existing Datasets")

datasets = automl.registry.list(type="dataset")


if datasets:
    for dataset in datasets:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"- **{dataset.name}** (ID: {dataset.id})")
        with col2:
            # Button to delete the dataset
            if st.button("Delete", key=f"delete_{dataset.id}"):
                datasets = automl.registry.list(type="dataset")
                automl.registry.delete(dataset.id)
                st.rerun()

        # Button to preview the dataset
        if st.toggle("Preview", key=f"preview_{dataset.id}"):
            st.subheader(f"Preview of '{dataset.name}'")
            try:
                preview_data = pd.read_csv(
                    os.path.join(".", "assets", "objects", dataset.asset_path)
                )
                st.dataframe(preview_data.head())
            except Exception as e:
                st.error(f"Failed to load dataset preview: {e}")
else:
    st.write("No datasets available.")
