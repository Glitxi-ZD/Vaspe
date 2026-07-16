import torch
import torch.nn as nn
import numpy as np


class SharedFrequencySelectiveMixer(nn.Module):
    def __init__(self, dim, order=3, rank=16, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.order = order
        self.rank = rank

        self.proj_layer = nn.Linear(dim, dim * (order + 1))

        self.shared_filters = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.01) for _ in range(order)
        ])

        self.freq_gates = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.01) for _ in range(order)
        ])

    def forward(self, x, edge_index):
        N = x.size(0)
        device = x.device

        proj_out = self.proj_layer(x)
        proj_splits = torch.split(proj_out, self.dim, dim=-1)

        V = proj_splits[-1]
        Ps = proj_splits[:-1]

        for i in range(self.order):
            filt = self.shared_filters[i].unsqueeze(0).expand(N, -1)

            freq_importance = torch.sigmoid(self.freq_gates[i])
            
            conv_out = Ps[i] * filt * freq_importance.unsqueeze(0)

            V = V * conv_out

        return V

    def get_frequency_responses(self, device=None):
        """获取每阶的共享频率响应（用于可视化）"""
        if device is None:
            device = next(self.parameters()).device
        responses = []
        for i in range(self.order):
            filt = self.shared_filters[i]
            f_fft = torch.fft.rfft(filt, dim=0)  # filt is 1D [dim]
            responses.append(f_fft.abs().detach().cpu().numpy())
        return responses


class StructureAwareFrequencySelectiveMixer(nn.Module):
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

            # 频域滤波（与 Personalized 版本相同：在节点维度dim=0上进行FFT）
            p_fft = torch.fft.rfft(Ps[i], dim=0, norm='ortho')
            f_fft = torch.fft.rfft(filt, dim=0, norm='ortho')

            n_freq = p_fft.size(0)
            freq_importance = torch.sigmoid(self.freq_gates[i])
            freq_decay = torch.exp(
                -torch.arange(n_freq, device=device, dtype=torch.float32) / (n_freq / 3 + 1e-8)
            ).unsqueeze(-1)

            conv_fft = p_fft * f_fft * freq_importance.unsqueeze(0) * freq_decay
            conv_out = torch.fft.irfft(conv_fft, n=N, dim=0, norm='ortho')

            V = V * conv_out

        return V

    def get_frequency_responses(self, edge_index, N, device, node_mask=None):
        struct_features = self._compute_structural_features(edge_index, N, device)
        struct_enc = self.struct_encoder(struct_features)
        pos_enc = self._positional_encoding(N, device)
        combined_enc = torch.cat([pos_enc, struct_enc], dim=-1)

        responses = []
        for i in range(self.order):
            filt = self.filter_generators[i](combined_enc)
            f_fft = torch.fft.rfft(filt, dim=0, norm='ortho')
            responses.append(f_fft.abs().detach().cpu().numpy())

        return responses, struct_features.detach().cpu()
