import streamlit as st
import pandas as pd

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
    st.write("Preview of the uploaded dataset:")
    st.write(df.head())

    # Convert the DataFrame to a Dataset artifact
    dataset_name = st.text_input("Enter a name for the dataset")
    if st.button("Save Dataset"):
        if dataset_name:
            dataset = Dataset.from_dataframe(df, name=dataset_name, asset_path=f"datasets/{dataset_name}.csv")
            automl.registry.register(dataset)
            st.success(f"Dataset '{dataset_name}' saved successfully!")
        else:
            st.error("Please enter a name for the dataset.")

# List existing datasets
st.subheader("Existing Datasets")
datasets = automl.registry.list(type="dataset")
if datasets:
    for dataset in datasets:
        st.write(f"- {dataset.name} (ID: {dataset.id})")
else:
    st.write("No datasets available.")
