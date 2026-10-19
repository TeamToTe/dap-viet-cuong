"""
Models module initialization.
Imports model plugins to self-register into MODEL_REGISTRY.
"""

from .tvae import TVAE
from .fence_gan import FenceGAN
from .tab_ddpm import TabDDPM

__all__ = ["TVAE", "FenceGAN", "TabDDPM"]
