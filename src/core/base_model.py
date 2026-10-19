"""
Abstract Base Class for Tabular Deep Generative Models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import torch
import numpy as np


class BaseGenerativeModel(ABC):
    """
    Abstract interface for all tabular anomaly generative models
    (TVAE, FenceGAN, TabDDPM, etc.).
    """
    plugin_name: str = "base_model"

    def __init__(self, input_dim: int, device: Optional[str] = None, **kwargs):
        self.input_dim = input_dim
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, X_train: np.ndarray, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, **kwargs) -> Dict[str, Any]:
        """
        Train the generative model on normal baseline sensor data.
        Returns training loss history / metadata.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate synthetic anomaly samples in standardized feature space.
        Returns array of shape (n_samples, input_dim).
        """
        pass

    def save(self, filepath: str) -> None:
        """Default PyTorch checkpoint save."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "kwargs": self.kwargs,
            "state_dict": getattr(self, "state_dict", lambda: {})()
        }
        torch.save(state, filepath)

    def load(self, filepath: str) -> None:
        """Default PyTorch checkpoint load."""
        if os.path.exists(filepath):
            state = torch.load(filepath, map_location=self.device)
            if hasattr(self, "load_state_dict") and "state_dict" in state:
                self.load_state_dict(state["state_dict"])
