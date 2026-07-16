import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveVAE(nn.Module):
    def __init__(self, latent_dim=8, feature_dim=384, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.component_gates = nn.Parameter(torch.ones(latent_dim))

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)
        z = mu + eps * std

        x_recon = self.decoder(z)

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        recon_loss = F.mse_loss(x_recon, x, reduction='mean') / self.feature_dim

        active_mask = torch.sigmoid(self.component_gates)
        weighted_z = z * active_mask.unsqueeze(0)
        weighted_z = F.normalize(weighted_z, p=2, dim=-1) * torch.sqrt(
            torch.tensor(self.latent_dim, device=z.device, dtype=z.dtype)
        )

        return weighted_z, (mu, std), kl_div, recon_loss, active_mask
