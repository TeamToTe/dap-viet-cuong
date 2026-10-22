"""
SKAB (Skoltech Anomaly Benchmark) Dataset Plugin.
Industrial testbed with 8 sensor variables from valve1 circulation loop.
Auto-downloads from GitHub repository waico/SKAB if not available locally.
"""

import os
import glob
import re
import urllib.request
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ...core.registry import register_dataset
from ...core.base_dataset import BaseDataset, TabularDataResult


def _run_id(path: str) -> int:
    """Extract the numeric run id from a file path ('run_07.csv' -> 7, '7.csv' -> 7)."""
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else 0


@register_dataset("skab")
class SKABDataset(BaseDataset):
    """
    SKAB testbed dataset plugin.
    Splits first 9 runs for Training (filtered strictly for anomaly == 0.0),
    and last 7 runs for Evaluation (held-out normal + real anomalies).
    """
    def __init__(self, data_dir: str = "./data", **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.skab_dir = os.path.join(data_dir, "raw", "skab", "valve1")
        self.skab_processed_dir = os.path.join(data_dir, "processed", "skab", "valve1")

    def _ensure_downloaded(self) -> List[str]:
        base_url = "https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/"
        # NOTE: only the raw directory is a download target - the processed directory
        # additionally contains aggregate files (all_runs.csv, run_index.csv) that
        # are not individual runs.
        candidates = [
            self.skab_dir,
            os.path.join("SKAB_data", "valve1"),
            os.path.join("Data_Repo", "SKAB", "data", "valve1")
        ]
        valve_files = []
        for path in candidates:
            files = sorted(glob.glob(os.path.join(path, "*.csv")), key=_run_id)
            if len(files) >= 4:
                return files

        os.makedirs(self.skab_dir, exist_ok=True)
        print(f"📦 Downloading SKAB Valve1 benchmark data to {self.skab_dir}...")
        for i in range(16):
            fn = f"{i}.csv"
            dest = os.path.join(self.skab_dir, fn)
            if not os.path.exists(dest):
                try:
                    urllib.request.urlretrieve(base_url + fn, dest)
                except Exception as e:
                    print(f"Warning: Failed downloading {fn}: {e}")
        valve_files = sorted(glob.glob(os.path.join(self.skab_dir, "*.csv")), key=_run_id)
        return valve_files

    def _discover_runs(self) -> Tuple[List[str], str]:
        """
        Locate the run files and the delimiter to read them with.

        Prefers the reformatted files in `data/processed/skab/valve1/` (comma-delimited,
        snake_case headers) and falls back to the original semicolon-delimited raw files.
        Returns (file_paths, separator) with files ordered by numeric run id.
        """
        processed = sorted(
            [p for p in glob.glob(os.path.join(self.skab_processed_dir, "run_*.csv"))
             if re.fullmatch(r"run_\d+\.csv", os.path.basename(p))],
            key=_run_id,
        )
        if len(processed) >= 4:
            return processed, ","

        raw = sorted(glob.glob(os.path.join(self.skab_dir, "*.csv")), key=_run_id)
        if len(raw) >= 4:
            return raw, ";"

        return self._ensure_downloaded(), ";"

    def load(self) -> TabularDataResult:
        files, sep = self._discover_runs()
        if not files:
            raise RuntimeError("Could not find or download SKAB CSV files")

        # Sort by numeric run id so the train/test split is 0-8 vs 9-15
        # (a plain lexicographic sort would interleave 10-15 between 1 and 2).
        files = sorted(files, key=_run_id)

        train_files = files[:9]
        test_files = files[9:]

        df_train = pd.concat(
            [pd.read_csv(f, sep=sep).assign(run=_run_id(f)) for f in train_files],
            ignore_index=True,
        )
        df_test = pd.concat(
            [pd.read_csv(f, sep=sep).assign(run=_run_id(f)) for f in test_files],
            ignore_index=True,
        )

        sensor_cols = [c for c in df_train.columns if c not in ['datetime', 'anomaly', 'changepoint', 'run']]

        # Train set: strictly normal samples (anomaly == 0.0)
        df_train_norm = df_train[df_train['anomaly'] == 0.0]

        X_train_raw = df_train_norm[sensor_cols].values
        X_test_raw = df_test[sensor_cols].values
        y_test = df_test['anomaly'].astype(int).values

        # Standardize strictly on Normal Training data
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        return TabularDataResult(
            X_train_norm=X_train_norm,
            X_test=X_test,
            y_test=y_test,
            feature_names=sensor_cols,
            scaler=scaler,
            dataset_name="SKAB_Valve1",
            raw_train_df=df_train_norm,
            raw_test_df=df_test
        )
