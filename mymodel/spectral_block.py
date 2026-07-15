"""
改进版谱域混合器块
保留块内局部传播，图结构信息由谱域混合器负责提取
"""
import torch
import torch.nn as nn

from .components import DataAdaptivePropagation, StructureAwareFrequencySelectiveMixer
from .shared_spectral_mixer import (
    SharedFrequencySelectiveMixer,
    StructureAwareFrequencySelectiveMixer as StructureAwareMixerPersonalized,
)


class ImprovedSpectralMixerBlock(nn.Module):
    """改进版谱域混合器块 - 保留块内局部传播"""
    def __init__(self, dim, order=2, rank=8, ff_mult=4, dropout=0.1, filter_mode='personalized'):
        super().__init__()
        self.order = order
        self.filter_mode = filter_mode

        self.local_prop = DataAdaptivePropagation(norm=True)
        self.norm_local = nn.LayerNorm(dim)

        if order > 0:
            if filter_mode == 'shared':
                self.mixer = SharedFrequencySelectiveMixer(dim, order, rank, dropout)
            else:
                self.mixer = StructureAwareMixerPersonalized(dim, order, rank, dropout)
            self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        local_feat = self.local_prop(x, edge_index)
        x = x + self.dropout(self.norm_local(local_feat))

        if self.order > 0 and hasattr(self, 'mixer'):
            mixer_out = self.mixer(x, edge_index)
            x = x + self.dropout(self.norm1(mixer_out))

        ffn_out = self.ffn(x)
        x = x + self.dropout(self.norm2(ffn_out))
        return x
