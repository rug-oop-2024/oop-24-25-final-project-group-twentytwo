import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoML system
automl = AutoMLSystem.get_instance()

# Page title
st.title("Dataset Management")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    if st.toggle("Preview"):
        st.write("Preview of the uploaded dataset:")
        st.write(df.head())

    # Convert the DataFrame to a Dataset artifact
    dataset_name = st.text_input("Enter a name for the dataset")
    base_path = automl._storage._base_path
    asset_path = os.path.join(base_path, f"{dataset_name}.csv")
    if st.button("Save Dataset"):
        if dataset_name:
            dataset = Dataset.from_dataframe(
                df,
                name=dataset_name,
                asset_path=f"datasets/{dataset_name}.csv",
            )
            automl.registry.register(dataset)
            st.write({asset_path})
            st.success(f"Dataset '{dataset_name}' saved successfully!")
        else:
            st.error("Please enter a name for the dataset.")

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
                # Call the registry's delete method
                # Refresh the dataset list
                datasets = automl.registry.list(type="dataset")
                automl.registry.delete(dataset.id)
                st.rerun()

        # Button to preview the dataset
        if st.toggle("Preview", key=f"preview_{dataset.id}"):
            st.subheader(f"Preview of '{dataset.name}'")
            try:
                # Use the asset path for loading the dataset
                preview_data = pd.read_csv(
                    os.path.join(".", "assets", "objects", dataset.asset_path)
                )
                st.dataframe(preview_data.head())  # Show first 5 rows
            except Exception as e:
                st.error(f"Failed to load dataset preview: {e}")
else:
    st.write("No datasets available.")
