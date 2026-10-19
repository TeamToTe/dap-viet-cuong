"""
Data module initialization.
Imports all dataset plugins so they self-register in DATASET_REGISTRY.
"""

from .datasets.skab import SKABDataset
from .datasets.ai4i import AI4IDataset
from .datasets.generic_csv import GenericCSVDataset

__all__ = ["SKABDataset", "AI4IDataset", "GenericCSVDataset"]
