"""
自适应变分自编码器（AdaptiveVAE）
替代GMM进行主题建模，输出接口与AdaptiveGMM完全兼容
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveVAE(nn.Module):
    """
    自适应变分自编码器：替代GMM进行主题建模
    输出接口与AdaptiveGMM完全兼容
    """
    def __init__(self, latent_dim=8, feature_dim=384, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 编码器: x -> (mu, log_var)
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

        # 解码器: z -> x_recon
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

        # 潜在门控：动态学习活跃维度
        self.component_gates = nn.Parameter(torch.ones(latent_dim))

    def forward(self, x):
        """
        Returns:
            weighted_z: [N, latent_dim] 门控后潜在变量
            (mu, std): 分布参数
            kl_div: KL散度
            recon_loss: 重构损失
            active_mask: 活跃维度掩码
        """
        # 编码
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)

        # 重参数化采样
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 解码重构
        x_recon = self.decoder(z)

        # KL散度
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        # 重构损失 - 按维度归一化
        recon_loss = F.mse_loss(x_recon, x, reduction='mean') / self.feature_dim

        # 软门控
        active_mask = torch.sigmoid(self.component_gates)
        weighted_z = z * active_mask.unsqueeze(0)
        weighted_z = F.normalize(weighted_z, p=2, dim=-1) * torch.sqrt(
            torch.tensor(self.latent_dim, device=z.device, dtype=z.dtype)
        )

        return weighted_z, (mu, std), kl_div, recon_loss, active_mask
