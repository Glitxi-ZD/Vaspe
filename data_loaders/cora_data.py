"""
Cora数据集加载模块
支持链接预测和节点分类双任务，包含边划分功能
使用 Planetoid 数据集
"""
import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, remove_self_loops
from typing import List, Tuple, Dict


class CoraLinkPredictionDataset:
    """
    Cora数据集 - 支持链接预测和节点分类双任务
    
    数据集统计:
    - 节点数: 2,708
    - 边数: 5,429
    - 特征维度: 1,433
    - 类别数: 7 (用于节点分类)
    """
    
    def __init__(self, data_path='./data', task='link_prediction', 
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        初始化Cora数据集
        
        Args:
            data_path: 数据存储路径
            task: 任务类型 ('link_prediction' 或 'node_classification')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        self.task = task
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载原始数据
        dataset = Planetoid(root=data_path, name='Cora')
        data = dataset[0]
        
        # 处理边（去除自环并转换为无向图）
        edge_index = remove_self_loops(data.edge_index)[0]
        self.full_edge_index = to_undirected(edge_index)
        
        # 节点特征（双任务共用）
        self.node_features = data.x
        self.num_nodes = data.num_nodes
        self.num_features = data.num_features
        
        # 节点分类特有属性
        self.num_classes = dataset.num_classes
        self.y = data.y
        
        # 设置分割
        if task == 'link_prediction':
            self._setup_link_prediction(train_ratio, val_ratio)
        else:
            self._setup_node_classification(train_ratio, val_ratio)
    
    def _setup_link_prediction(self, train_ratio, val_ratio):
        """设置链接预测所需的边分割"""
        # 获取所有边（转换为元组列表）
        edges = []
        seen = set()
        for i in range(self.full_edge_index.shape[1]):
            src = int(self.full_edge_index[0, i])
            dst = int(self.full_edge_index[1, i])
            # 只保留一个方向（无向图）
            edge_key = (min(src, dst), max(src, dst))
            if edge_key not in seen and src != dst:
                seen.add(edge_key)
                edges.append(edge_key)
        
        # 随机打乱
        random.shuffle(edges)
        
        # 计算划分数量
        n_edges = len(edges)
        n_train = int(n_edges * train_ratio)
        n_val = int(n_edges * val_ratio)
        n_test = n_edges - n_train - n_val
        
        # 划分边集
        self.train_edges = edges[:n_train]
        self.val_edges = edges[n_train:n_train + n_val]
        self.test_edges = edges[n_train + n_val:]
        
        print(f"Cora - 链接预测边划分:")
        print(f"  总边数: {n_edges}")
        print(f"  训练集: {len(self.train_edges)} ({len(self.train_edges)/n_edges*100:.1f}%)")
        print(f"  验证集: {len(self.val_edges)} ({len(self.val_edges)/n_edges*100:.1f}%)")
        print(f"  测试集: {len(self.test_edges)} ({len(self.test_edges)/n_edges*100:.1f}%)")
    
    def _setup_node_classification(self, train_ratio, val_ratio):
        """设置节点分类所需的节点分割"""
        indices = torch.randperm(self.num_nodes)
        n_train = int(self.num_nodes * train_ratio)
        n_val = int(self.num_nodes * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        
        print(f"Cora - 节点分类节点划分:")
        print(f"  总节点数: {self.num_nodes}")
        print(f"  训练集: {n_train} ({n_train/self.num_nodes*100:.1f}%)")
        print(f"  验证集: {n_val} ({n_val/self.num_nodes*100:.1f}%)")
        print(f"  测试集: {len(test_idx)} ({len(test_idx)/self.num_nodes*100:.1f}%)")
    
    def get_train_edge_index(self):
        """
        获取训练边索引 [2, E] (双向边)
        用于图卷积网络的消息传递
        """
        if self.task == 'link_prediction':
            edges = self.train_edges
            edge_list = []
            for src, dst in edges:
                edge_list.extend([[src, dst], [dst, src]])
            
            if len(edge_list) == 0:
                return torch.zeros((2, 0), dtype=torch.long)
            
            return torch.tensor(edge_list, dtype=torch.long).t()
        else:
            # 节点分类时使用全图
            return self.full_edge_index
    
    def generate_negative_samples(self, num_negatives=100, split='test'):
        """
        生成负样本（仅用于链接预测）
        
        Args:
            num_negatives: 每个正样本对应的负样本数
            split: 数据集划分 ('train', 'val', 'test')
        
        Returns:
            {src_node: [neg_dst1, neg_dst2, ...]}
        """
        if self.task != 'link_prediction':
            raise ValueError("generate_negative_samples only available for link_prediction task")
        
        # 获取对应的数据集边
        if split == 'train':
            pos_edges = self.train_edges
        elif split == 'val':
            pos_edges = self.val_edges
        elif split == 'all':
            pos_edges = self.train_edges + self.val_edges + self.test_edges
        else:
            pos_edges = self.test_edges
        
        # 构建已存在的边集合（用于过滤）
        existing_edges = set()
        for src, dst in self.train_edges + self.val_edges + self.test_edges:
            existing_edges.add((src, dst))
            existing_edges.add((dst, src))
        
        neg_samples = {}
        for src, dst in pos_edges:
            if src not in neg_samples:
                neg_samples[src] = []
            
            attempts = 0
            while len(neg_samples[src]) < num_negatives and attempts < num_negatives * 10:
                neg_dst = random.randint(0, self.num_nodes - 1)
                if neg_dst != src and (src, neg_dst) not in existing_edges:
                    neg_samples[src].append(neg_dst)
                attempts += 1
        
        return neg_samples
    
    def get_split_indices(self):
        """
        获取节点分类的分割索引
        
        Returns:
            {'train': train_indices, 'val': val_indices, 'test': test_indices}
        """
        if self.task != 'node_classification':
            raise ValueError("get_split_indices only available for node_classification task")
        
        return {
            'train': self.train_mask.nonzero(as_tuple=True)[0],
            'val': self.val_mask.nonzero(as_tuple=True)[0],
            'test': self.test_mask.nonzero(as_tuple=True)[0]
        }
    
    def get_statistics(self):
        """
        获取数据集统计信息
        
        Returns:
            dict: 包含各种统计指标的字典
        """
        stats = {
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'task': self.task,
        }
        
        if self.task == 'link_prediction':
            stats.update({
                'num_train_edges': len(self.train_edges),
                'num_val_edges': len(self.val_edges),
                'num_test_edges': len(self.test_edges),
                'total_edges': len(self.train_edges) + len(self.val_edges) + len(self.test_edges),
            })
        else:
            stats.update({
                'num_train_nodes': self.train_mask.sum().item(),
                'num_val_nodes': self.val_mask.sum().item(),
                'num_test_nodes': self.test_mask.sum().item(),
            })
        
        return stats


def create_cora_dataset(root='./data', task='link_prediction', 
                        train_ratio=0.8, val_ratio=0.1, seed=42, **kwargs):
    """
    创建Cora数据集
    
    此函数兼容 train_link_prediction.py 和 train_node_classification.py 的调用方式
    
    Args:
        root: 数据存储路径
        task: 任务类型 ('link_prediction' 或 'node_classification')
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        **kwargs: 其他参数（忽略）
    
    Returns:
        CoraLinkPredictionDataset 实例
    """
    return CoraLinkPredictionDataset(
        data_path=root,
        task=task,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )


# 保持向后兼容的类
class CoraDataset:
    """Cora数据集类（向后兼容）"""
    def __init__(self, data_path='./data'):
        dataset = create_cora_dataset(root=data_path)
        self.data = dataset
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
    
    def get_data(self):
        return self.data


class CoraNodeClassificationDataset(CoraLinkPredictionDataset):
    """Cora节点分类数据集（向后兼容）"""
    def __init__(self, data_path='./data', train_ratio=0.6, val_ratio=0.2, seed=42):
        super().__init__(data_path=data_path, task='node_classification',
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
