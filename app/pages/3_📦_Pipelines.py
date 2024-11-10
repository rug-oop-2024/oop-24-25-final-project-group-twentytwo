import streamlit as st
from app.core.system import AutoMLSystem

st.set_page_config(page_title="Deployment", page_icon="📦")

st.write("# 📦 Deployment")
st.write("View and manage saved pipelines.")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")

if not pipelines:
    st.info("No pipelines have been saved yet.")
else:
    pipeline_names = [
        f"{pipeline.name} v{pipeline.version}" for pipeline in pipelines
    ]
    selected_pipeline_name = st.selectbox(
        "Select a pipeline to view", pipeline_names
    )

    selected_pipeline = next(
        pipeline
        for pipeline in pipelines
        if f"{pipeline.name} v{pipeline.version}" == selected_pipeline_name
    )

    st.subheader(
        f"Pipeline: {selected_pipeline.name} v{selected_pipeline.version}"
    )
    st.write("**Name:**", selected_pipeline.name)
    st.write("**Version:**", selected_pipeline.version)
    st.write("**Path:**", selected_pipeline.asset_path)
