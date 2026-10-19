"""
Metrics module initialization.
Imports metric plugins so they register into METRIC_REGISTRY.
"""

from .fidelity import FidelityMetric
from .correlation import CorrelationMetric
from .utility import DownstreamUtilityMetric

__all__ = ["FidelityMetric", "CorrelationMetric", "DownstreamUtilityMetric"]
