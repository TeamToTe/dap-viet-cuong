"""
Base Dataset Class and Data Structures for Industrial Tabular Data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class TabularDataResult:
    """Container for preprocessed industrial tabular dataset."""
    X_train_norm: np.ndarray          # 100% normal operational data for training generative models
    X_test: np.ndarray                # Test sensor features (contains both normal and anomalies)
    y_test: np.ndarray                # Ground truth test labels: 0 = Normal, 1 = Anomaly
    feature_names: List[str]          # Sensor feature column names
    scaler: Any                       # Fitted scaler object (e.g. StandardScaler)
    dataset_name: str                 # Identifier for the dataset
    raw_train_df: Optional[pd.DataFrame] = None
    raw_test_df: Optional[pd.DataFrame] = None

    @property
    def input_dim(self) -> int:
        return self.X_train_norm.shape[1]

    @property
    def real_anomalies_test(self) -> np.ndarray:
        return self.X_test[self.y_test == 1]

    @property
    def real_normals_test(self) -> np.ndarray:
        return self.X_test[self.y_test == 0]


class BaseDataset(ABC):
    """Abstract Base Class for Dataset Plugins."""
    plugin_name: str = "base"

    def __init__(self, data_dir: str = "./data", **kwargs):
        self.data_dir = data_dir
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> TabularDataResult:
        """
        Loads, preprocesses, and splits dataset.
        Returns:
            TabularDataResult containing standardized normal train data and held-out test data.
        """
        pass
