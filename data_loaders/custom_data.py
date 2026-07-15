"""
Custom数据集加载模块 - 本地化版本
从本地文件加载，不需要连接数据库
"""
import torch
import numpy as np
import random
import os
from torch_geometric.utils import to_undirected, remove_self_loops
from typing import List, Tuple, Dict


class CustomDataset:
    """
    Custom数据集 (学术论文引用网络)
    
    数据集统计:
    - 节点数: 25,678
    - 边数: 5,667 (去重后)
    - 特征维度: 384 (Sentence-BERT文本嵌入)
    - 划分方式: 按时间 (train≤2023, val=2024, test>2024)
    """
    
    def __init__(self, data_path='./data/custom', task='link_prediction',
                 train_ratio=0.8, val_ratio=0.1, seed=42, split_mode='time'):
        """
        初始化Custom数据集
        split_mode: 'time' (按时间划分) 或 'random' (随机划分)
        """
        self.task = task
        self.seed = seed
        self.data_path = data_path
        self.split_mode = split_mode
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据
        self._load_data()
        
        # 设置分割
        if task == 'link_prediction':
            self._setup_link_prediction()
        else:
            self._setup_node_classification(train_ratio, val_ratio)
    
    def _load_data(self):
        """从本地文件加载数据"""
        print(f"Loading Custom dataset from local files...")
        
        data_file = os.path.join(self.data_path, 'processed', 'data.pt')
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Custom dataset not found at {data_file}\n"
                f"Please run extract_custom_dataset.py first to extract from database."
            )
        
        data = torch.load(data_file, map_location='cpu')
        
        # 提取数据
        self.node_features = data.x
        self.num_nodes = data.x.size(0)
        self.num_features = data.x.size(1)
        
        # 处理边
        edge_index = remove_self_loops(data.edge_index)[0]
        self.full_edge_index = to_undirected(edge_index)
        
        # 存储额外信息
        self.paper_ids = data.paper_ids if hasattr(data, 'paper_ids') else None
        self.paper_years = data.paper_years if hasattr(data, 'paper_years') else None
        
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Features: {self.num_features}")
        print(f"  Total edges: {self.full_edge_index.size(1) // 2}")
    
    def _setup_link_prediction(self):
        """设置链接预测划分"""
        if self.split_mode == 'random':
            # 随机划分：合并所有边 → 随机打乱 → 70/15/15
            train_file = os.path.join(self.data_path, 'train_edges.npy')
            val_file = os.path.join(self.data_path, 'val_edges.npy')
            test_file = os.path.join(self.data_path, 'test_edges.npy')
            
            all_edges = []
            for f in [train_file, val_file, test_file]:
                all_edges.extend([tuple(e) for e in np.load(f).tolist()])
            
            random.shuffle(all_edges)
            n = len(all_edges)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            
            self.train_edges = all_edges[:n_train]
            self.val_edges = all_edges[n_train:n_train + n_val]
            self.test_edges = all_edges[n_train + n_val:]
            
            print(f"Custom - Link Prediction Split (random):")
        else:
            # 时间划分：从文件加载
            train_file = os.path.join(self.data_path, 'train_edges.npy')
            val_file = os.path.join(self.data_path, 'val_edges.npy')
            test_file = os.path.join(self.data_path, 'test_edges.npy')
            
            if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
                raise FileNotFoundError("Edge split files not found.")
            
            self.train_edges = [tuple(e) for e in np.load(train_file).tolist()]
            self.val_edges = [tuple(e) for e in np.load(val_file).tolist()]
            self.test_edges = [tuple(e) for e in np.load(test_file).tolist()]
            
            print(f"Custom - Link Prediction Split (time-based):")
        
        print(f"  Train: {len(self.train_edges)}, Val: {len(self.val_edges)}, Test: {len(self.test_edges)}")
    
    def _setup_node_classification(self, train_ratio, val_ratio):
        """设置节点分类分割"""
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
        
        print(f"Custom - Node Classification Split:")
        print(f"  Total: {self.num_nodes}, Train: {n_train}, Val: {n_val}, Test: {len(test_idx)}")
    
    def get_train_edge_index(self):
        """获取训练边索引"""
        if self.task == 'link_prediction':
            edges = self.train_edges
            edge_list = []
            for src, dst in edges:
                edge_list.extend([[src, dst], [dst, src]])
            
            if len(edge_list) == 0:
                return torch.zeros((2, 0), dtype=torch.long)
            
            return torch.tensor(edge_list, dtype=torch.long).t()
        else:
            return self.full_edge_index
    
    def generate_negative_samples(self, num_negatives=100, split='test'):
        """生成负样本"""
        if self.task != 'link_prediction':
            raise ValueError("Only for link_prediction")
        
        if split == 'train':
            pos_edges = self.train_edges
        elif split == 'val':
            pos_edges = self.val_edges
        elif split == 'all':
            pos_edges = self.train_edges + self.val_edges + self.test_edges
        else:
            pos_edges = self.test_edges
        
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
        """获取分割索引"""
        if self.task != 'node_classification':
            raise ValueError("Only for node_classification")
        
        return {
            'train': self.train_mask.nonzero(as_tuple=True)[0],
            'val': self.val_mask.nonzero(as_tuple=True)[0],
            'test': self.test_mask.nonzero(as_tuple=True)[0]
        }
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'task': self.task,
        }
        
        if self.task == 'link_prediction':
            stats.update({
                'num_train_edges': len(self.train_edges),
                'num_val_edges': len(self.val_edges),
                'num_test_edges': len(self.test_edges),
            })
        else:
            stats.update({
                'num_train_nodes': self.train_mask.sum().item(),
                'num_val_nodes': self.val_mask.sum().item(),
                'num_test_nodes': self.test_mask.sum().item(),
            })
        
        return stats


def create_custom_dataset(root='./data', task='link_prediction',
                          train_ratio=0.8, val_ratio=0.1, seed=42, split_mode='time', **kwargs):
    """创建Custom数据集"""
    data_path = os.path.join(root, 'custom')
    
    return CustomDataset(
        data_path=data_path,
        task=task,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        split_mode=split_mode
    )


# 向后兼容
class CustomLinkPredictionDataset(CustomDataset):
    def __init__(self, data_path='./data/custom', train_ratio=0.8, val_ratio=0.1, seed=42):
        super().__init__(data_path=data_path, task='link_prediction',
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


class CustomNodeClassificationDataset(CustomDataset):
    def __init__(self, data_path='./data/custom', train_ratio=0.6, val_ratio=0.2, seed=42):
        super().__init__(data_path=data_path, task='node_classification',
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
