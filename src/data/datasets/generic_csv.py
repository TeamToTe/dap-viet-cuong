"""
Generic CSV Dataset Plugin.
Enables running the entire pipeline on ANY new industrial tabular dataset
simply by providing a CSV path, target column, and optional drop columns.
"""

import os
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ...core.registry import register_dataset
from ...core.base_dataset import BaseDataset, TabularDataResult


@register_dataset("generic_csv")
class GenericCSVDataset(BaseDataset):
    """
    Plugin for any generic industrial anomaly detection CSV file.
    Filter rule: Normal data (target == normal_val) is strictly used for training generative models.
    """
    def __init__(
        self,
        csv_path: str = "",
        target_col: str = "anomaly",
        normal_val: int = 0,
        drop_cols: Optional[List[str]] = None,
        test_size: float = 0.35,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self.target_col = target_col
        self.normal_val = normal_val
        self.drop_cols = drop_cols or []
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> TabularDataResult:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

        # Auto-detect separator
        try:
            df = pd.read_csv(self.csv_path, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(self.csv_path)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in CSV. Columns: {list(df.columns)}")

        # Drop unwanted identifier or timestamp columns
        drop_list = [c for c in self.drop_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in drop_list and c != self.target_col]

        # Handle numeric conversion
        for col in feature_cols:
            if df[col].dtype == object or str(df[col].dtype) == 'category':
                df[col] = pd.factorize(df[col])[0]

        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Labels: 0 = Normal, 1 = Anomaly
        y = (df[self.target_col] != self.normal_val).astype(int).values
        X = df[feature_cols].values

        # Stratified train/test split
        X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Industrial semi-supervised setup: Train strictly on NORMAL instances
        train_norm_mask = (y_train_raw == 0)
        X_train_norm_raw = X_train_raw[train_norm_mask]

        # Standardize strictly on Normal Training data
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_norm_raw)
        X_test = scaler.transform(X_test_raw)

        return TabularDataResult(
            X_train_norm=X_train_norm,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            scaler=scaler,
            dataset_name=os.path.splitext(os.path.basename(self.csv_path))[0],
            raw_train_df=df.iloc[train_norm_mask],
            raw_test_df=None
        )
