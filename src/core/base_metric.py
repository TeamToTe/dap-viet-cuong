"""
Abstract Base Class for Metric Plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseMetric(ABC):
    """Abstract interface for benchmark metric plugins."""
    plugin_name: str = "base_metric"

    @abstractmethod
    def evaluate(
        self,
        real_norm: np.ndarray,
        real_anom: np.ndarray,
        synth_anom: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate synthetic anomalies against real test data.
        Returns a dictionary of scalar metrics or DataFrames.
        """
        pass
