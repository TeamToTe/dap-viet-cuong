"""
Core Registry and Base Classes for Everything-is-a-Plugin Architecture.
"""

from .registry import (
    register_dataset,
    register_model,
    register_metric,
    get_dataset,
    get_model,
    get_metric,
    list_datasets,
    list_models,
    list_metrics,
)
from .base_dataset import BaseDataset, TabularDataResult
from .base_model import BaseGenerativeModel
from .base_metric import BaseMetric

__all__ = [
    "register_dataset",
    "register_model",
    "register_metric",
    "get_dataset",
    "get_model",
    "get_metric",
    "list_datasets",
    "list_models",
    "list_metrics",
    "BaseDataset",
    "TabularDataResult",
    "BaseGenerativeModel",
    "BaseMetric",
]
