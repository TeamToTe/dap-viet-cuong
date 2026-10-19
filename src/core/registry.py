"""
Plugin Registries for Datasets, Models, and Evaluation Metrics.
Supports extensible plugin architecture: simply decorate with @register_*
"""

from typing import Dict, Type, Any, Callable
import logging

logger = logging.getLogger(__name__)

DATASET_REGISTRY: Dict[str, Type] = {}
MODEL_REGISTRY: Dict[str, Type] = {}
METRIC_REGISTRY: Dict[str, Type] = {}


def register_dataset(name: str) -> Callable[[Type], Type]:
    """Decorator to register a new dataset plugin."""
    def decorator(cls: Type) -> Type:
        key = name.lower().strip()
        if key in DATASET_REGISTRY:
            logger.warning(f"Dataset '{key}' is being re-registered. Overwriting.")
        DATASET_REGISTRY[key] = cls
        cls.plugin_name = key
        return cls
    return decorator


def register_model(name: str) -> Callable[[Type], Type]:
    """Decorator to register a new generative model plugin."""
    def decorator(cls: Type) -> Type:
        key = name.lower().strip()
        if key in MODEL_REGISTRY:
            logger.warning(f"Model '{key}' is being re-registered. Overwriting.")
        MODEL_REGISTRY[key] = cls
        cls.plugin_name = key
        return cls
    return decorator


def register_metric(name: str) -> Callable[[Type], Type]:
    """Decorator to register a new evaluation metric plugin."""
    def decorator(cls: Type) -> Type:
        key = name.lower().strip()
        if key in METRIC_REGISTRY:
            logger.warning(f"Metric '{key}' is being re-registered. Overwriting.")
        METRIC_REGISTRY[key] = cls
        cls.plugin_name = key
        return cls
    return decorator


def get_dataset(name: str, **kwargs) -> Any:
    """Instantiate a dataset plugin by name."""
    key = name.lower().strip()
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not found in registry. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[key](**kwargs)


def get_model(name: str, **kwargs) -> Any:
    """Instantiate a model plugin by name."""
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def get_metric(name: str, **kwargs) -> Any:
    """Instantiate a metric plugin by name."""
    key = name.lower().strip()
    if key not in METRIC_REGISTRY:
        raise KeyError(
            f"Metric '{name}' not found in registry. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[key](**kwargs)


def list_datasets() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def list_metrics() -> list[str]:
    return list(METRIC_REGISTRY.keys())
