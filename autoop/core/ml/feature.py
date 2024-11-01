from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    # attributes here

    def __str__(self):
        raise NotImplementedError("To be implemented.")
