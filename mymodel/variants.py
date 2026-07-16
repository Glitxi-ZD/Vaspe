from .base_model import VAEBaseModel


class VAEFullModel(VAEBaseModel):
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=True, **kwargs)


class VAENoSpectralModel(VAEBaseModel):
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=False, **kwargs)


class VAENoVAEModel(VAEBaseModel):
    def __init__(self, **kwargs):
        super().__init__(use_vae=False, use_spectral=True, **kwargs)


class VAENoVAENoSpectralModel(VAEBaseModel):
    def __init__(self, **kwargs):
        super().__init__(use_vae=False, use_spectral=False, **kwargs)


class SharedFilterModel(VAEBaseModel):
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=True, filter_mode='shared', **kwargs)


class PersonalizedFilterModel(VAEBaseModel):
    def __init__(self, **kwargs):
        super().__init__(use_vae=True, use_spectral=True, filter_mode='personalized', **kwargs)


VAE_MODELS = {
    'vae_full': VAEFullModel,
    'vae_no_spectral': VAENoSpectralModel,
    'vae_no_vae': VAENoVAEModel,
    'vae_no_vae_no_spectral': VAENoVAENoSpectralModel,
    'shared_filter': SharedFilterModel,
    'personalized_filter': PersonalizedFilterModel,
}
