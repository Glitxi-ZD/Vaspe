"""
mymodel - VAE消融实验模型包
包含：AdaptiveVAE、谱域混合器块、基础组件、4种消融变体
"""
from .components import (
    DataAdaptivePropagation,
    StructureAwareFrequencySelectiveMixer,
    AdaptiveFeatureFusion,
)
from .vae import AdaptiveVAE
from .spectral_block import ImprovedSpectralMixerBlock
from .base_model import VAEBaseModel
from .variants import (
    VAEFullModel,
    VAENoSpectralModel,
    VAENoVAEModel,
    VAENoVAENoSpectralModel,
    VAE_MODELS
)

__all__ = [
    'DataAdaptivePropagation',
    'StructureAwareFrequencySelectiveMixer',
    'AdaptiveFeatureFusion',
    'AdaptiveVAE',
    'ImprovedSpectralMixerBlock',
    'VAEBaseModel',
    'VAEFullModel',
    'VAENoSpectralModel',
    'VAENoVAEModel',
    'VAENoVAENoSpectralModel',
    'VAE_MODELS',
]
