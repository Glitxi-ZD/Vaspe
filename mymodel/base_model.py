"""
VAE消融实验模型基类 - 门控融合版
核心修改：
1. 图通道入口使用门控融合：自适应权衡原始特征 vs VAE潜变量
2. 删除基类自带的local_prop和local_gate
3. 图结构信息完全由谱域混合器负责提取
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import AdaptiveFeatureFusion, DataAdaptivePropagation
from .shared_spectral_mixer import (
    SharedFrequencySelectiveMixer,
    StructureAwareFrequencySelectiveMixer,
)
from .vae import AdaptiveVAE
from .spectral_block import ImprovedSpectralMixerBlock


class VAEBaseModel(nn.Module):
    """
    VAE版消融实验模型基类 - 门控融合版
    
    门控融合机制：
        gate = sigmoid(Linear(concat([x, vae_z])))
        h = gate * x_proj + (1 - gate) * z_proj
        
    效果：模型自适应学习"什么时候看原始特征，什么时候看主题"
    """
    def __init__(
        self,
        latent_dim=8,
        input_dim=384,
        hidden_dim=64,
        output_dim=64,
        num_blocks=2,
        order=2,
        rank=8,
        dropout=0.4,
        ff_mult=4,
        use_spectral=True,
        use_vae=True,
        num_classes=None,
        filter_mode='personalized'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_spectral = use_spectral
        self.use_vae = use_vae
        self.num_classes = num_classes
        self.filter_mode = filter_mode

        # 文本分支
        self.text_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # VAE分支（可选）
        if use_vae:
            self.vae = AdaptiveVAE(latent_dim, input_dim, hidden_dim)
            # 门控融合：原始特征和VAE潜变量分别投影到hidden_dim
            self.x_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.z_proj = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            # 门控层：输入为concat([x, vae_z])，输出标量门控值
            self.gate_layer = nn.Linear(input_dim + latent_dim, 1)
        else:
            self.vae = None
            self.z_proj = None
            self.gate_layer = None
            # 无VAE时，用MLP将原始特征映射到32维（压缩表示）
            mlp_dim = 32
            self.x_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.GELU(),
            )
            self.mlp_dim = mlp_dim
            # 从32维投影回hidden_dim，供谱域混合器使用
            self.mlp_to_hidden = nn.Linear(mlp_dim, hidden_dim)

        # 谱域混合器（可选）
        if use_spectral:
            self.blocks = nn.ModuleList([
                ImprovedSpectralMixerBlock(hidden_dim, order, rank, ff_mult=ff_mult, dropout=dropout, filter_mode=filter_mode)
                for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList()

        # 图分支输出
        self.graph_out = nn.Linear(hidden_dim, output_dim)

        # 自适应特征融合
        self.adaptive_fusion = AdaptiveFeatureFusion(output_dim)

        # 节点分类头
        if num_classes is not None:
            self.classifier = nn.Linear(output_dim, num_classes)
        else:
            self.classifier = None

    def encode(self, x, edge_index, debug=False):
        """编码器 - 门控融合版本"""
        N = x.size(0)
        device = x.device

        # 1. 文本分支
        text_embed = self.text_mlp(x)

        # 2. 图分支（门控融合 or 纯特征）
        if self.use_vae and self.vae is not None:
            vae_features, (_, _), kl_div, recon_loss, _ = self.vae(x)
            vae_loss = {'kl_loss': kl_div, 'entropy_loss': recon_loss}

            # 门控融合
            x_proj = self.x_proj(x)                    # [N, hidden_dim]
            z_proj = self.z_proj(vae_features)          # [N, hidden_dim]
            gate_input = torch.cat([x, vae_features], dim=-1)  # [N, input_dim + latent_dim]
            gate = torch.sigmoid(self.gate_layer(gate_input))  # [N, 1]
            h = gate * x_proj + (1 - gate) * z_proj    # [N, hidden_dim]
        else:
            vae_loss = {'kl_loss': None, 'entropy_loss': None}
            mlp_feat = self.x_proj(x)  # [N, 32]
            h = self.mlp_to_hidden(mlp_feat)  # [N, hidden_dim]

        # 谱域混合器
        for block in self.blocks:
            h = block(h, edge_index)

        # 图分支输出
        graph_embed = self.graph_out(h)

        # 3. 自适应融合
        deg = torch.zeros(N, device=device)
        if edge_index.numel() > 0:
            deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
            deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
        deg = deg.clamp(min=1)

        final_embed, _ = self.adaptive_fusion(text_embed, graph_embed, deg)

        return final_embed, vae_loss

    def forward(self, x, edge_index, return_embeddings=False, debug=False):
        """前向传播，支持链接预测和节点分类双任务"""
        embeddings, vae_loss = self.encode(x, edge_index, debug)

        if return_embeddings or self.classifier is None:
            return embeddings, vae_loss

        logits = self.classifier(embeddings)
        return logits, embeddings, vae_loss
