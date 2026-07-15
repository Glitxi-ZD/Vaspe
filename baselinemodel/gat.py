"""
GAT (Graph Attention Network) 基线模型
纯PyTorch实现，支持节点分类和链接预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    """标准GAT卷积层（多头注意力）"""
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.3):
        super().__init__()
        self.heads = heads
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """前向传播"""
        N = x.size(0)
        device = x.device
        
        # 线性变换
        h = self.W(x)  # [N, out_channels]
        h = h.view(N, self.heads, self.head_dim)  # [N, heads, head_dim]
        
        # 计算注意力系数
        # src: a_src * h_i, dst: a_dst * h_j
        attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # [N, heads]
        attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [N, heads]
        
        # 注意力分数
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        e = self.leaky_relu(attn_src[src_idx] + attn_dst[dst_idx])  # [E, heads]
        
        # 归一化
        # 使用scatter实现softmax
        max_e = torch.zeros(N, self.heads, device=device)
        max_e.index_reduce_(0, dst_idx, e, 'amax', include_self=False)
        max_e = max_e.index_select(0, dst_idx)
        
        e = torch.exp(e - max_e)
        
        # 归一化
        sum_e = torch.zeros(N, self.heads, device=device)
        sum_e.index_add_(0, dst_idx, e)
        sum_e = sum_e.index_select(0, dst_idx)
        
        attn = e / (sum_e + 1e-8)
        attn = self.dropout(attn)
        
        # 聚合
        h_src = h[src_idx]  # [E, heads, head_dim]
        
        # 加权聚合
        out = torch.zeros(N, self.heads, self.head_dim, device=device)
        out.index_add_(0, dst_idx, h_src * attn.unsqueeze(-1))
        
        return out.view(N, -1)  # [N, out_channels]


class GATBaseline(nn.Module):
    """GAT基线，支持节点分类"""
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, 
                 num_layers=2, dropout=0.3, num_classes=None, heads=8):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.heads = heads
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 节点分类头
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
        """前向传播"""
        embeddings = self.encode(x, edge_index)
        
        if return_embeddings or self.classifier is None:
            return embeddings, None
        
        # 节点分类模式
        logits = self.classifier(embeddings)
        return logits, embeddings, None
