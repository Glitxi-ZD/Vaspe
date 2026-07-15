"""
简化的 GPM (Graph Pattern Mining) 模型核心实现
基于 GPM 的核心思想：使用向量量化学习图模式
优化版本：移除复杂的链接编码，专注于节点表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizeSimple(nn.Module):
    """
    简化的向量量化模块
    """
    def __init__(self, dim, codebook_size=32, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        
        # Codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        
        # EMA
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', self.codebook.data.clone())
        
    def forward(self, x):
        """
        Args:
            x: [N, dim] 输入特征
        Returns:
            quantized: [N, dim] 量化后的特征
            commit_loss: commitment loss
        """
        # 计算与codebook的距离
        dist = torch.sum((x.unsqueeze(1) - self.codebook) ** 2, dim=2)  # [N, codebook_size]
        
        # 找到最近的codebook向量
        indices = torch.argmin(dist, dim=1)  # [N]
        
        # 量化
        quantized = self.codebook[indices]  # [N, dim]
        
        # EMA更新
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()  # [N, codebook_size]
                
                # 更新聚类大小
                cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                
                # 更新codebook
                emb_sum = encodings.t() @ x
                self.ema_w.mul_(self.decay).add_(emb_sum, alpha=1 - self.decay)
                
                # 重新归一化
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                self.codebook.data = self.ema_w / cluster_size.unsqueeze(1)
        
        # Commitment loss
        commit_loss = F.mse_loss(x, quantized.detach())
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, commit_loss


class GPMSimple(nn.Module):
    """
    简化的 GPM 模型
    核心思想：使用 VQ 学习图模式 + GNN 传播
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3,
                 codebook_size=32, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 向量量化层
        self.vq = VectorQuantizeSimple(hidden_dim, codebook_size=codebook_size)
        
        # GNN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.LayerNorm(hidden_dim))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def propagate(self, x, edge_index):
        """简单的消息传播"""
        row, col = edge_index
        
        # 聚合邻居特征
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row])
        
        # 归一化
        deg = torch.bincount(col, minlength=x.size(0)).float().unsqueeze(1).clamp(min=1)
        out = out / deg
        
        return out
        
    def forward(self, x, edge_index):
        """
        前向传播
        Args:
            x: [N, input_dim] 节点特征
            edge_index: [2, E] 边索引
        Returns:
            embed: [N, output_dim] 节点嵌入
            vq_loss: 向量量化损失
        """
        # 输入投影
        h = self.input_proj(x)
        
        # 向量量化
        h_vq, vq_loss = self.vq(h)
        
        # 融合原始特征和量化特征
        h = h + h_vq
        
        # GNN层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # 消息传播
            h_neigh = self.propagate(h, edge_index)
            
            # 更新
            h = conv(h_neigh)
            h = bn(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 输出投影
        embed = self.output_proj(h)
        
        # 归一化
        embed = F.normalize(embed, p=2, dim=1)
        
        return embed, vq_loss


class GPMSimpleAdapter(nn.Module):
    """
    简化 GPM 适配器，支持节点分类
    """
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3, 
                 codebook_size=32, num_classes=None, **kwargs):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        self.model = GPMSimple(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            codebook_size=codebook_size
        )
        
        # 节点分类头（如果指定了num_classes）
        if num_classes is not None:
            self.classifier = nn.Linear(output_dim, num_classes)
        else:
            self.classifier = None
        
    def encode(self, x, edge_index):
        """编码器：获取节点嵌入"""
        embed, vq_loss = self.model(x, edge_index)
        
        if vq_loss is None:
            vq_loss = torch.tensor(0.0, device=x.device)
        
        # 返回损失字典以兼容训练代码
        loss_dict = {'vq_loss': vq_loss}
        
        return embed, loss_dict
        
    def forward(self, x, edge_index, return_embeddings=False, debug=False):
        """
        前向传播，支持节点分类
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            return_embeddings: 是否返回embeddings（链接预测模式）
            debug: 是否打印调试信息
            
        Returns:
            如果return_embeddings=True或classifier为None:
                (embeddings, loss_dict) - 用于链接预测
            否则:
                (logits, embeddings, loss_dict) - 用于节点分类
        """
        # 获取嵌入
        embeddings, loss_dict = self.encode(x, edge_index)
        
        # 如果指定了return_embeddings或没有分类头，返回embeddings（链接预测兼容模式）
        if return_embeddings or self.classifier is None:
            return embeddings, loss_dict
        
        # 节点分类模式
        logits = self.classifier(embeddings)
        return logits, embeddings, loss_dict


# 注册到 baseline_adapters 中
def register_gpm_simple():
    """注册简化版 GPM"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from baseline_adapters import BASELINE_MODELS
        BASELINE_MODELS['gpm_simple'] = GPMSimpleAdapter
        print("[INFO] Registered GPM Simple adapter")
    except Exception as e:
        print(f"[WARNING] Failed to register GPM Simple: {e}")


if __name__ == '__main__':
    # 测试
    model = GPMSimpleAdapter(input_dim=1433, hidden_dim=64, output_dim=64)
    x = torch.randn(2708, 1433)
    edge_index = torch.randint(0, 2708, (2, 5000))
    
    embed, loss = model(x, edge_index)
    print(f"Output shape: {embed.shape}")
    print(f"VQ loss: {loss}")
