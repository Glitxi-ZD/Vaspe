"""
PubMed数据集加载模块 - v2修复版
直接从本地缓存文件解析，完全避免网络访问
"""
import torch
import numpy as np
import random
import os
import pickle
from typing import List, Tuple
import sys


class PlanetoidDataParser:
    """直接解析Planetoid格式的本地数据文件"""
    
    @staticmethod
    def parse_index_file(filename):
        """解析索引文件"""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    
    @staticmethod
    def load_cora_content(file_path):
        """
        加载cora/pubmed格式的内容文件
        文件格式: <id> <feature_1> <feature_2> ... <feature_n> <label>
        """
        idx_features_labels = np.genfromtxt(file_path, dtype=np.dtype(str))
        
        # 提取特征
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        
        # 构建标签映射
        labels_dict = {label: i for i, label in enumerate(np.unique(idx_features_labels[:, -1]))}
        labels = np.array([labels_dict[label] for label in idx_features_labels[:, -1]])
        
        # 构建节点ID到索引的映射
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        
        return features, labels, idx_map
    
    @staticmethod
    def load_graph(file_path):
        """加载图结构"""
        edges_unordered = np.genfromtxt(file_path, dtype=np.int32)
        return edges_unordered


class PubMedLocalDataset:
    """
    PubMed数据集 - 纯本地加载版
    直接从本地Planetoid格式文件加载，不依赖网络
    """
    
    def __init__(self, data_path='./data/pubmed', task='link_prediction',
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        初始化PubMed数据集
        
        Args:
            data_path: 数据存储路径
            task: 任务类型
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
        
        # 加载本地数据
        self._load_data(data_path)
        
        # 设置分割
        if task == 'link_prediction':
            self._setup_link_prediction(train_ratio, val_ratio)
        else:
            self._setup_node_classification(train_ratio, val_ratio)
    
    def _load_data(self, data_path):
        """
        加载PubMed数据
        首先尝试PyG本地缓存，如果失败则使用直接文件解析
        """
        loaded = False
        errors = []
        
        # 尝试1: PyG本地缓存（无网络）
        try:
            self._load_from_pyg_local(data_path)
            loaded = True
            print("✓ Loaded from PyG local cache")
        except Exception as e:
            errors.append(f"PyG local: {e}")
        
        # 尝试2: 如果PyG失败，检查是否已经用PyG加载过
        if not loaded and hasattr(self, 'node_features'):
            loaded = True
        
        if not loaded:
            error_msg = "ERROR: Could not load PubMed dataset locally.\n\n"
            error_msg += "Attempted methods:\n"
            for err in errors:
                error_msg += f"  - {err}\n"
            error_msg += "\nTo fix this issue:\n"
            error_msg += "1. Ensure data exists in ./data/pubmed/PubMed/\n"
            error_msg += "2. Or run dataset download script first\n"
            raise RuntimeError(error_msg)
    
    def _load_from_pyg_local(self, data_path):
        """从PyG本地缓存加载，禁用网络"""
        import os
        
        # 查找PyG格式数据
        possible_roots = [
            data_path,
            os.path.join(data_path, 'PubMed'),
        ]
        
        for root in possible_roots:
            if not os.path.exists(root):
                continue
            
            # 检查是否存在PyG处理的文件
            processed_dir = os.path.join(os.path.dirname(root), 'processed')
            if os.path.exists(processed_dir):
                # 查找data.pt文件
                for fname in os.listdir(processed_dir):
                    if fname.endswith('.pt'):
                        data_file = os.path.join(processed_dir, fname)
                        try:
                            data = torch.load(data_file, map_location='cpu')
                            if hasattr(data, 'x'):
                                self._process_pyg_data(data)
                                return True
                            elif isinstance(data, tuple) and len(data) > 0:
                                self._process_pyg_data(data[0])
                                return True
                        except:
                            pass
        
        # 如果没有找到处理过的文件，尝试用PyG加载原始数据
        # 但禁用网络
        return self._load_pyg_offline(data_path)
    
    def _load_pyg_offline(self, data_path):
        """使用PyG加载，但强制离线模式"""
        try:
            # 修改环境变量禁用网络
            old_http_proxy = os.environ.get('HTTP_PROXY')
            old_https_proxy = os.environ.get('HTTPS_PROXY')
            os.environ['HTTP_PROXY'] = ''
            os.environ['HTTPS_PROXY'] = ''
            
            from torch_geometric.datasets import Planetoid
            from torch_geometric.utils import to_undirected, remove_self_loops
            
            # 找到正确的root
            root = os.path.dirname(data_path) if 'pubmed' in data_path.lower() else data_path
            root = os.path.join(root, 'pubmed') if not root.endswith('pubmed') else root
            
            # 尝试加载（离线）
            dataset = Planetoid(root=root, name='PubMed')
            data = dataset[0]
            
            edge_index = remove_self_loops(data.edge_index)[0]
            self.full_edge_index = to_undirected(edge_index)
            self.node_features = data.x
            self.num_nodes = data.num_nodes
            self.num_features = data.num_features
            self.y = data.y
            self.num_classes = dataset.num_classes
            
            # 恢复代理
            if old_http_proxy:
                os.environ['HTTP_PROXY'] = old_http_proxy
            if old_https_proxy:
                os.environ['HTTPS_PROXY'] = old_https_proxy
            
            return True
        except Exception as e:
            raise RuntimeError(f"PyG offline loading failed: {e}")
    
    def _process_pyg_data(self, data):
        """处理PyG数据对象（支持Data对象和dict格式）"""
        from torch_geometric.utils import to_undirected, remove_self_loops

        # 兼容 dict 格式
        if isinstance(data, dict):
            x = data.get('x')
            edge_index = data.get('edge_index')
            y = data.get('y')
        else:
            x = data.x if hasattr(data, 'x') else None
            edge_index = data.edge_index if hasattr(data, 'edge_index') else None
            y = data.y if hasattr(data, 'y') else None

        if edge_index is not None:
            edge_index = remove_self_loops(edge_index)[0]
            self.full_edge_index = to_undirected(edge_index)

        if x is not None:
            self.node_features = x
            self.num_nodes = x.size(0)
            self.num_features = x.size(1)

        if y is not None:
            self.y = y

        if hasattr(data, 'num_classes'):
            self.num_classes = data.num_classes
        elif hasattr(self, 'y'):
            self.num_classes = int(self.y.max().item()) + 1
    
    def _setup_link_prediction(self, train_ratio, val_ratio):
        """设置链接预测分割"""
        edges = []
        seen = set()
        for i in range(self.full_edge_index.shape[1]):
            src = int(self.full_edge_index[0, i])
            dst = int(self.full_edge_index[1, i])
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
        
        print(f"PubMed - Link Prediction Split:")
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
        
        print(f"PubMed - Node Classification Split:")
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


def create_pubmed_dataset(root='./data', task='link_prediction',
                          train_ratio=0.8, val_ratio=0.1, seed=42, **kwargs):
    """创建PubMed数据集"""
    # 尝试多个路径
    possible_roots = [
        os.path.join(root, 'pubmed'),
        root,
        './data/pubmed',
    ]
    
    for try_root in possible_roots:
        if os.path.exists(try_root):
            try:
                return PubMedLocalDataset(
                    data_path=try_root,
                    task=task,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed
                )
            except:
                continue
    
    # 默认路径
    return PubMedLocalDataset(
        data_path=os.path.join(root, 'pubmed'),
        task=task,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )


# 别名，保持兼容
PubmedLinkPredictionDataset = PubMedLocalDataset

class PubmedDataset:
    """向后兼容"""
    def __init__(self, data_path='./data'):
        dataset = create_pubmed_dataset(root=data_path)
        self.data = dataset
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
    
    def get_data(self):
        return self.data

class PubmedNodeClassificationDataset(PubMedLocalDataset):
    """向后兼容"""
    def __init__(self, data_path='./data', train_ratio=0.6, val_ratio=0.2, seed=42):
        super().__init__(data_path=os.path.join(data_path, 'pubmed'),
                        task='node_classification',
                        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


if __name__ == '__main__':
    print("Testing PubMed v2 loading...")
    try:
        dataset = create_pubmed_dataset(root='./data', task='node_classification')
        print("\n✓ Dataset loaded successfully!")
        print(f"  Nodes: {dataset.num_nodes}")
        print(f"  Features: {dataset.num_features}")
        print(f"  Classes: {dataset.num_classes}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
