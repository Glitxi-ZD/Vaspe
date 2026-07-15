"""
mymodel 组件
包含谱域混合器和特征融合所需的基础模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DataAdaptivePropagation(nn.Module):
    """
    数据自适应局部传播：基于全局度数分布自适应阈值
    复杂度：O(E + N)
    """
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm
        self.threshold_logits = nn.Parameter(torch.tensor(0.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index
        device = x.device

        agg = torch.zeros_like(x)
        agg = agg.scatter_add(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        agg = agg.scatter_add(0, src.unsqueeze(1).expand(-1, x.size(1)), x[dst])

        deg = torch.zeros(N, device=device)
        deg = deg.scatter_add(0, src, torch.ones_like(src, dtype=torch.float))
        deg = deg.scatter_add(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = deg.clamp(min=1)

        if self.norm:
            agg = agg / deg.sqrt().unsqueeze(1)

        median_deg = deg.median()
        adaptive_threshold = median_deg * torch.sigmoid(self.threshold_logits)
        gate = torch.sigmoid((deg - adaptive_threshold) / (self.temperature.abs() + 0.1))
        agg = agg * gate.unsqueeze(1)

        return agg


class StructureAwareFrequencySelectiveMixer(nn.Module):
    """
    结构感知频率选择谱域混合器
    - 利用图结构信息指导滤波器生成
    - 频率选择性：区分高频/低频分量
    - 多阶递归更新机制
    复杂度：O(N log N + E)
    """
    def __init__(self, dim, order=3, rank=16, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.order = order
        self.rank = rank

        self.proj_layer = nn.Linear(dim, dim * (order + 1))

        self.struct_encoder = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.filter_generators = nn.ModuleList()
        for _ in range(order):
            generator = nn.Sequential(
                nn.Linear(dim * 2, dim * rank),
                nn.LayerNorm(dim * rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * rank, dim)
            )
            self.filter_generators.append(generator)

        self.freq_gates = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.01) for _ in range(order)
        ])

        self._cached_struct_features = None
        self._cached_edge_index = None

    def _compute_structural_features(self, edge_index, N, device):
        src, dst = edge_index
        deg = torch.zeros(N, device=device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        local_density = deg / N
        deg_norm = deg / (deg.max() + 1e-8)
        return torch.stack([deg_norm, local_density], dim=-1)

    def _positional_encoding(self, N, device):
        position = torch.arange(N, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device) *
            (-np.log(10000.0) / self.dim)
        )
        pos_enc = torch.zeros(N, self.dim, device=device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x, edge_index):
        N = x.size(0)
        device = x.device

        proj_out = self.proj_layer(x)
        proj_splits = torch.split(proj_out, self.dim, dim=-1)

        V = proj_splits[-1]
        Ps = proj_splits[:-1]

        if self.training or self._cached_struct_features is None or \
           (self._cached_edge_index is not None and not torch.equal(edge_index, self._cached_edge_index)):
            struct_features = self._compute_structural_features(edge_index, N, device)
            struct_enc = self.struct_encoder(struct_features)
            if not self.training:
                self._cached_struct_features = struct_enc
                self._cached_edge_index = edge_index.clone()
        else:
            struct_enc = self._cached_struct_features

        pos_enc = self._positional_encoding(N, device)
        combined_enc = torch.cat([pos_enc, struct_enc], dim=-1)

        for i in range(self.order):
            filt = self.filter_generators[i](combined_enc)

            p_fft = torch.fft.rfft(Ps[i], dim=0)
            f_fft = torch.fft.rfft(filt, dim=0)

            n_freq = p_fft.size(0)
            freq_importance = torch.sigmoid(self.freq_gates[i])
            freq_decay = torch.exp(
                -torch.arange(n_freq, device=device) / (n_freq / 3)
            ).unsqueeze(-1)

            conv_fft = p_fft * f_fft * freq_importance.unsqueeze(0) * freq_decay
            conv_out = torch.fft.irfft(conv_fft, n=N, dim=0)

            V = V * conv_out

        return V


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合：为每个节点学习最佳alpha权重
    复杂度：O(Nd)
    """
    def __init__(self, output_dim):
        super().__init__()
        self.alpha_predictor = nn.Sequential(
            nn.Linear(output_dim * 2 + 1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, text_embed, graph_embed, deg):
        deg_norm = (deg / (deg.max() + 1e-8)).unsqueeze(-1)
        combined = torch.cat([text_embed, graph_embed, deg_norm], dim=-1)
        alpha = torch.sigmoid(self.alpha_predictor(combined))
        final_embed = alpha * text_embed + (1 - alpha) * graph_embed
        return F.normalize(final_embed, p=2, dim=-1), alpha.squeeze(-1)
