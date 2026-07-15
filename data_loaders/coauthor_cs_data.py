"""
Coauthor CS数据集加载模块 - VPN友好版
支持通过网络下载（使用系统代理/VPN）
"""
import torch
import numpy as np
import random
import os
from torch_geometric.utils import to_undirected, remove_self_loops
from typing import List, Tuple, Dict


class CoauthorCSDataset:
    """
    Coauthor CS数据集
    
    数据集统计:
    - 节点数: 18,333
    - 边数: 163,788
    - 特征维度: 6,805
    - 类别数: 15 (研究领域)
    """
    
    def __init__(self, data_path='./data', task='link_prediction',
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        初始化Coauthor CS数据集
        """
        self.task = task
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据
        self._load_data(data_path)
        
        # 设置分割
        if task == 'link_prediction':
            self._setup_link_prediction(train_ratio, val_ratio)
        else:
            self._setup_node_classification(train_ratio, val_ratio)
    
    def _load_data(self, data_path):
        """
        加载Coauthor CS数据
        首先检查本地缓存，如果不存在再尝试网络下载
        """
        print(f"Loading Coauthor CS dataset...")
        
        # 先尝试本地缓存（快速路径）
        try:
            self._load_from_local_cache(data_path)
            print(f"  Successfully loaded from local cache")
            return
        except Exception as e:
            print(f"  Local cache not found, trying network download...")
        
        # 如果本地缓存不存在，尝试网络下载
        try:
            self._load_from_network(data_path)
            print(f"  Successfully downloaded/loaded Coauthor CS")
            return
        except Exception as e:
            error_msg = f"""
            ERROR: Could not load Coauthor CS dataset!
            
            Tried:
            1. Local cache
            2. Network download (may need VPN/proxy)
            
            Error: {e}
            """
            raise RuntimeError(error_msg)
    
    def _load_from_network(self, data_path):
        """通过网络下载（使用系统代理设置）"""
        import socket
        
        old_timeout = socket.getdefaulttimeout()
        
        try:
            socket.setdefaulttimeout(30)  # 30秒超时，避免长时间阻塞
            
            from torch_geometric.datasets import Coauthor
            
            root = data_path
            if not os.path.exists(root):
                os.makedirs(root, exist_ok=True)
            
            print(f"  Downloading from network (using system proxy if configured)...")
            dataset = Coauthor(root=root, name='CS')
            data = dataset[0]
            self._process_loaded_data(data, dataset.num_classes)
            
        finally:
            socket.setdefaulttimeout(old_timeout)
    
    def _load_from_local_cache(self, data_path):
        """从本地缓存加载"""
        possible_paths = [
            os.path.join(data_path, 'CS'),
            os.path.join(data_path, 'Coauthor', 'CS'),
            data_path,
            './data/CS',
            './data/Coauthor/CS',
        ]
        
        for path in possible_paths:
            if not os.path.exists(path):
                continue
            
            processed_dir = os.path.join(path, 'processed')
            if os.path.exists(processed_dir):
                for fname in os.listdir(processed_dir):
                    if fname.endswith('.pt'):
                        data_file = os.path.join(processed_dir, fname)
                        try:
                            data = torch.load(data_file, map_location='cpu')
                            if hasattr(data, 'x') or (isinstance(data, (list, tuple)) and len(data) > 0):
                                self._process_loaded_data(data)
                                return
                        except Exception as e:
                            print(f"    Failed to load {fname}: {e}")
                            continue
        
        raise RuntimeError("No local cache found")
    
    def _process_loaded_data(self, data, num_classes=None):
        """处理加载的数据"""
        from torch_geometric.utils import to_undirected, remove_self_loops
        
        # Coauthor CS .pt 文件格式可能是 (dict, None, Data) 元组
        if isinstance(data, (tuple, list)) and len(data) > 0:
            # 检查是否是 (dict, None, Data) 格式
            if isinstance(data[0], dict):
                data_dict = data[0]
                # 从 dict 中提取 edge_index
                if 'edge_index' in data_dict:
                    edge_index = data_dict['edge_index']
                    self.full_edge_index = to_undirected(edge_index)
                else:
                    raise ValueError("Data dict missing edge_index")
                # 从 dict 中提取 features
                if 'x' in data_dict:
                    self.node_features = data_dict['x']
                    self.num_nodes = self.node_features.size(0)
                    self.num_features = self.node_features.size(1)
                else:
                    raise ValueError("Data dict missing features (x)")
                # 从 dict 中提取 labels
                if 'y' in data_dict:
                    self.y = data_dict['y']
                else:
                    raise ValueError("Data dict missing labels (y)")
                self.num_classes = num_classes if num_classes is not None else int(self.y.max().item()) + 1
                print(f"  - Nodes: {self.num_nodes:,}")
                print(f"  - Features: {self.num_features}")
                print(f"  - Classes: {self.num_classes}")
                return
            else:
                data = data[0]
        
        if hasattr(data, 'edge_index'):
            edge_index = remove_self_loops(data.edge_index)[0]
            self.full_edge_index = to_undirected(edge_index)
        else:
            raise ValueError("Data missing edge_index")
        
        if hasattr(data, 'x'):
            self.node_features = data.x
            self.num_nodes = data.x.size(0)
            self.num_features = data.x.size(1)
        else:
            raise ValueError("Data missing features")
        
        if hasattr(data, 'y'):
            self.y = data.y
        else:
            raise ValueError("Data missing labels")
        
        if num_classes is not None:
            self.num_classes = num_classes
        elif hasattr(data, 'num_classes'):
            self.num_classes = data.num_classes
        else:
            self.num_classes = int(self.y.max().item()) + 1
        
        print(f"  - Nodes: {self.num_nodes:,}")
        print(f"  - Features: {self.num_features}")
        print(f"  - Classes: {self.num_classes}")
    
    def _setup_link_prediction(self, train_ratio, val_ratio):
        """设置链接预测分割"""
        edges = []
        seen = set()
        edge_index_np = self.full_edge_index.cpu().numpy()
        
        for i in range(edge_index_np.shape[1]):
            src = int(edge_index_np[0, i])
            dst = int(edge_index_np[1, i])
            edge_key = (min(src, dst), max(src, dst))
            if edge_key not in seen and src != dst:
                seen.add(edge_key)
                edges.append(edge_key)
        
        random.shuffle(edges)
        
        n_edges = len(edges)
        n_train = int(n_edges * train_ratio)
        n_val = int(n_edges * val_ratio)
        
        self.train_edges = edges[:n_train]
        self.val_edges = edges[n_train:n_train + n_val]
        self.test_edges = edges[n_train + n_val:]
        
        print(f"Coauthor CS - Link Prediction Split:")
        print(f"  Total: {n_edges}, Train: {len(self.train_edges)}, Val: {len(self.val_edges)}, Test: {len(self.test_edges)}")
    
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
        
        print(f"Coauthor CS - Node Classification Split:")
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
            'num_classes': self.num_classes,
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


def create_coauthor_cs_dataset(root='./data', task='link_prediction',
                               train_ratio=0.8, val_ratio=0.1, seed=42, **kwargs):
    """创建Coauthor CS数据集"""
    possible_roots = [
        root,
        os.path.join(root, 'CS'),
        os.path.join(root, 'Coauthor', 'CS'),
        './data/CS',
        './data/Coauthor/CS',
        './data',
    ]
    
    for try_root in possible_roots:
        try:
            return CoauthorCSDataset(
                data_path=try_root,
                task=task,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed
            )
        except:
            continue
    
    return CoauthorCSDataset(
        data_path=root,
        task=task,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )


# 向后兼容
class CoauthorCSLinkPredictionDataset(CoauthorCSDataset):
    def __init__(self, data_path='./data', train_ratio=0.8, val_ratio=0.1, seed=42):
        super().__init__(data_path=data_path, task='link_prediction',
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


class CoauthorCSNodeClassificationDataset(CoauthorCSDataset):
    def __init__(self, data_path='./data', train_ratio=0.6, val_ratio=0.2, seed=42):
        super().__init__(data_path=data_path, task='node_classification',
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
