"""
GCN (Graph Convolutional Network) 基线模型
纯PyTorch实现，支持链接预测和节点分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """标准GCN卷积层（向量化优化版）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        """前向传播"""
        N = x.size(0)
        device = x.device
        
        # 线性变换
        x = self.linear(x)
        
        # 计算度
        deg = torch.zeros(N, device=device)
        deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
        deg = deg.clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        
        # 归一化
        x_norm = x * deg_inv_sqrt.unsqueeze(1)
        
        # 聚合
        out = torch.zeros_like(x)
        edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        out.index_add_(0, edge_index[1], x_norm[edge_index[0]] * edge_weight.unsqueeze(1))
        out.index_add_(0, edge_index[0], x_norm[edge_index[1]] * edge_weight.unsqueeze(1))
        
        return out


class GCNBaseline(nn.Module):
    """GCN基线，支持节点分类"""
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3, num_classes=None):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 节点分类头（如果指定了num_classes）
        if num_classes is not None:
            self.classifier = nn.Linear(output_dim, num_classes)
        else:
            self.classifier = None
        
    def encode(self, x, edge_index):
        """编码器"""
        h = self.input_proj(x)
        h = F.gelu(h)
        
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # 残差连接
        
        embed = self.output_proj(h)
        return embed
        
    def forward(self, x, edge_index, return_embeddings=False, debug=False):
        """前向传播，支持节点分类"""
        embeddings = self.encode(x, edge_index)
        
        # 如果指定了return_embeddings或没有分类头，返回embeddings（链接预测兼容模式）
        if return_embeddings or self.classifier is None:
            return embeddings, None
        
        # 节点分类模式
        logits = self.classifier(embeddings)
        return logits, embeddings, None
    
    def _compute_degree(self, edge_index, N):
        """计算节点度数"""
        deg = torch.zeros(N, device=edge_index.device)
        deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=edge_index.device))
        return deg.clamp(min=1)
