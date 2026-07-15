"""
VAE消融实验模型变体
4种组合：VAE×谱域混合器

新增：Shared vs. Personalized Spectral Filter 实验变体
"""
from .base_model import VAEBaseModel


class VAEFullModel(VAEBaseModel):
    """VAE + 谱域混合器（完整模型）"""
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=True, **kwargs)


class VAENoSpectralModel(VAEBaseModel):
    """仅VAE，无谱域混合器"""
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=False, **kwargs)


class VAENoVAEModel(VAEBaseModel):
    """无VAE，仅谱域混合器 - 使用原始特征"""
    def __init__(self, **kwargs):
        super().__init__(use_vae=False, use_spectral=True, **kwargs)


class VAENoVAENoSpectralModel(VAEBaseModel):
    """无VAE且无谱域混合器 - 仅原始特征+MLP"""
    def __init__(self, **kwargs):
        super().__init__(use_vae=False, use_spectral=False, **kwargs)


# ============================================================================
# Shared vs. Personalized Spectral Filter 实验变体
# ============================================================================

class SharedFilterModel(VAEBaseModel):
    """
    Shared Spectral Filter 版本
    - 所有节点共享同一个可学习的频谱响应
    - 保留完整模型骨架（VAE、Gate、Refinement、Fusion、Prediction Head）
    - 仅将 spectral filter 生成方式改为共享
    """
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=True, filter_mode='shared', **kwargs)


class PersonalizedFilterModel(VAEBaseModel):
    """
    Personalized Spectral Filter 版本（默认）
    - 每个节点根据结构统计生成自己的频率响应
    - s_i = [degree_i, local_density_i]
    - g_i(ω) = MLP(s_i)
    - 与 Shared Filter 版本形成对照
    """
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=True, filter_mode='personalized', **kwargs)


# 模型注册表
VAE_MODELS = {
    'vae_full': VAEFullModel,
    'vae_no_spectral': VAENoSpectralModel,
    'vae_no_vae': VAENoVAEModel,
    'vae_no_vae_no_spectral': VAENoVAENoSpectralModel,
    # Shared vs. Personalized 实验变体
    'shared_filter': SharedFilterModel,
    'personalized_filter': PersonalizedFilterModel,
}
