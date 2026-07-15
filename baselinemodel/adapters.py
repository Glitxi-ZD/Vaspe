"""
Baseline 模型适配器
统一接口，支持创建和管理所有 baseline 模型
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入本包内的模型
from .gcn import GCNBaseline
from .graphsage import GraphSAGEBaseline
from .gpm_simple import GPMSimpleAdapter

# 可选导入完整版
try:
    from .gpm_full import GPMFullAdapter
    FULL_GPM_AVAILABLE = True
except ImportError:
    FULL_GPM_AVAILABLE = False

try:
    from .scnode_full import SCNodeFullAdapter
    FULL_SCNODE_AVAILABLE = True
except ImportError:
    FULL_SCNODE_AVAILABLE = False


class GPMAdapter(nn.Module):
    """
    GPM 模型适配器
    根据 use_full 参数选择简化版或完整版
    """
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3, 
                 use_full=False, num_classes=None, **kwargs):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        if use_full and FULL_GPM_AVAILABLE:
            print("[INFO] Using full GPM implementation")
            self.model = GPMFullAdapter(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
            self.is_full = True
        else:
            print("[INFO] Using simplified GPM implementation")
            self.model = GPMSimpleAdapter(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes,
                **kwargs
            )
            self.is_full = False
        
        # 节点分类头
        if num_classes is not None and not self.is_full:
            self.classifier = nn.Linear(output_dim, num_classes)
        else:
            self.classifier = None
    
    def encode(self, x, edge_index):
        """编码器"""
        return self.model.encode(x, edge_index)
    
    def forward(self, x, edge_index, return_embeddings=False, debug=False):
        """前向传播"""
        embeddings, loss_dict = self.encode(x, edge_index)
        
        if return_embeddings or self.classifier is None:
            return embeddings, loss_dict
        
        logits = self.classifier(embeddings)
        return logits, embeddings, loss_dict


class SCNodeAdapter(nn.Module):
    """
    SCNode 模型适配器
    根据 use_full 参数选择简化版或完整版
    """
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3,
                 use_full=False, num_classes=None, **kwargs):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        if use_full and FULL_SCNODE_AVAILABLE:
            print("[INFO] Using full SCNode implementation")
            self.model = SCNodeFullAdapter(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes,
                **kwargs
            )
            self.is_full = True
        else:
            print("[INFO] Using simplified SCNode implementation")
            self._init_simplified(input_dim, hidden_dim, output_dim, num_layers, dropout)
            self.is_full = False
        
        # 节点分类头
        if num_classes is not None and not self.is_full:
            self.classifier = nn.Linear(output_dim, num_classes)
        else:
            self.classifier = None
    
    def _init_simplified(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        """初始化简化版SCNode"""
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
    
    def encode(self, x, edge_index):
        """编码器"""
        if self.is_full:
            return self.model.encode(x, edge_index)
        
        # 简化版 - 使用度归一化的图卷积
        h = self.input_proj(x)
        h = F.gelu(h)
        
        row, col = edge_index
        # 计算双向度数
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
        deg = deg.clamp(min=1)  # 避免除零
        deg_inv_sqrt = deg.pow(-0.5).unsqueeze(1)
        
        for conv, bn in zip(self.convs, self.bns):
            h_norm = h * deg_inv_sqrt
            out = torch.zeros_like(h)
            out.scatter_add_(0, col.unsqueeze(1).expand_as(h_norm[row]), h_norm[row])
            out = out * deg_inv_sqrt
            
            h = h + conv(out)
            h = bn(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        embed = self.output_proj(h)
        # 确保没有 NaN
        embed = torch.nan_to_num(embed, nan=0.0)
        return embed, None
    
    def forward(self, x, edge_index, return_embeddings=False, debug=False):
        """前向传播"""
        embeddings, loss_dict = self.encode(x, edge_index)
        
        if return_embeddings or self.classifier is None:
            return embeddings, loss_dict
        
        logits = self.classifier(embeddings)
        return logits, embeddings, loss_dict


# 模型注册表
BASELINE_REGISTRY = {
    'gcn': GCNBaseline,
    'graphsage': GraphSAGEBaseline,
    'gpm_simple': GPMSimpleAdapter,
    'gpm': GPMAdapter,
    'scnode': SCNodeAdapter,
}

if FULL_GPM_AVAILABLE:
    BASELINE_REGISTRY['gpm_full'] = GPMFullAdapter
if FULL_SCNODE_AVAILABLE:
    BASELINE_REGISTRY['scnode_full'] = SCNodeFullAdapter


def create_model(model_name, input_dim=384, hidden_dim=64, output_dim=64, 
                 num_classes=None, **kwargs):
    """
    创建 baseline 模型的统一接口
    
    Args:
        model_name: 模型名称
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_classes: 节点分类类别数（None表示链接预测）
        **kwargs: 其他参数
    
    Returns:
        模型实例
    """
    if model_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_REGISTRY.keys())}")
    
    model_class = BASELINE_REGISTRY[model_name]
    
    return model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_classes=num_classes,
        **kwargs
    )
