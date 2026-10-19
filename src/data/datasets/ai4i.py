"""
AI4I 2020 Predictive Maintenance Dataset Plugin.
Auto-downloads from UCI Machine Learning Repository or Github mirrors.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ...core.registry import register_dataset
from ...core.base_dataset import BaseDataset, TabularDataResult


@register_dataset("ai4i")
class AI4IDataset(BaseDataset):
    """
    AI4I 2020 Predictive Maintenance Dataset plugin.
    Sensors: Air temperature, Process temperature, Rotational speed, Torque, Tool wear.
    Target: Machine failure (0 = Normal, 1 = Failure/Anomaly).
    """
    def __init__(self, data_dir: str = "./data", test_size: float = 0.3, random_state: int = 42, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.csv_path = os.path.join(data_dir, "raw", "ai4i2020.csv")
        self.test_size = test_size
        self.random_state = random_state

    def _ensure_downloaded(self) -> str:
        candidates = [
            self.csv_path,
            os.path.join(self.data_dir, "raw", "ai4i", "ai4i2020.csv"),
            os.path.join("data", "raw", "ai4i", "ai4i2020.csv"),
            os.path.join("data", "raw", "ai4i2020.csv"),
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                return p

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        # Try primary URL (archive mirror on raw github)
        urls = [
            "https://raw.githubusercontent.com/marshmellow77/ai4i2020/master/ai4i2020.csv",
            "https://archive.ics.uci.edu/static/public/601/data.csv"
        ]
        print(f"📦 Downloading AI4I 2020 dataset to {self.csv_path}...")
        downloaded = False
        for url in urls:
            try:
                urllib.request.urlretrieve(url, self.csv_path)
                if os.path.getsize(self.csv_path) > 1000:
                    downloaded = True
                    break
            except Exception as e:
                print(f"Notice: Failed from {url}: {e}")

        if not downloaded:
            raise RuntimeError(f"Could not download AI4I 2020 dataset automatically into {self.csv_path}")

        return self.csv_path

    def load(self) -> TabularDataResult:
        csv_file = self._ensure_downloaded()
        df = pd.read_csv(csv_file)

        target_col = "Machine failure"
        if target_col not in df.columns:
            for alt in ["anomaly", "failure", "Target"]:
                if alt in df.columns:
                    target_col = alt
                    break

        # Select sensor numeric features
        feature_candidates = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
        sensor_cols = [c for c in feature_candidates if c in df.columns]
        if not sensor_cols:
            # Fallback to all numeric columns except id and failure flags
            exclude = ["UDI", "Product ID", "Type", target_col, "TWF", "HDF", "PWF", "OSF", "RNF"]
            sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        # Extract X and y
        X = df[sensor_cols].values
        y = df[target_col].astype(int).values

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
            feature_names=sensor_cols,
            scaler=scaler,
            dataset_name="AI4I_Predictive_Maintenance",
            raw_train_df=None,
            raw_test_df=None
        )
