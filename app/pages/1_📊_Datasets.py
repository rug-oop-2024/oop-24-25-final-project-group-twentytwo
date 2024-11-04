import streamlit as st
import pandas as pd
import os
from datetime import datetime

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here