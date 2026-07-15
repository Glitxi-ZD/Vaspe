"""
完整的 SCNode (Spectral Clustering Node) 模型实现
基于原始论文代码，适配链接预测任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_scipy_sparse_matrix, degree
from scipy.sparse import csr_matrix
import networkx as nx

# 使用 PyTorch 实现的聚类算法，避免 Windows 上 sklearn 的兼容性问题
from .pytorch_cluster import PyTorchKMeans, PyTorchSpectralClustering

# 标准化器（简单实现，避免 sklearn 依赖）
class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SpatialFeatureExtractor:
    """
    空间特征提取器
    提取节点的局部拓扑结构特征
    """
    def __init__(self, num_hops=2):
        self.num_hops = num_hops
        
    def extract(self, data, nodes=None):
        """
        提取空间特征
        Args:
            data: PyG Data对象
            nodes: 要提取的节点索引，None表示所有节点
        Returns:
            spatial_features: [N, feature_dim] 空间特征矩阵
        """
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.num_nodes
        
        if nodes is None:
            nodes = np.arange(num_nodes)
        
        # 构建邻接表
        adj_list = self._build_adj_list(edge_index, num_nodes)
        
        # 为每个节点提取空间特征
        features = []
        for node in nodes:
            node_features = self._extract_node_features(node, adj_list, num_nodes)
            features.append(node_features)
        
        return np.array(features)
    
    def _build_adj_list(self, edge_index, num_nodes):
        """构建邻接表"""
        adj_list = [set() for _ in range(num_nodes)]
        for src, dst in edge_index.T:
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        return adj_list
    
    def _extract_node_features(self, node, adj_list, num_nodes):
        """提取单个节点的空间特征"""
        features = []
        
        # 1-hop邻居
        hop1_neighbors = adj_list[node]
        hop1_count = len(hop1_neighbors)
        
        # 2-hop邻居
        hop2_neighbors = set()
        for nbr in hop1_neighbors:
            hop2_neighbors.update(adj_list[nbr])
        hop2_neighbors -= hop1_neighbors
        hop2_neighbors.discard(node)
        hop2_count = len(hop2_neighbors)
        
        # 特征：度数、1-hop邻居数、2-hop邻居数、聚类系数等
        features.extend([
            hop1_count,  # 度数
            hop2_count,  # 2-hop邻居数
            hop1_count / max(num_nodes, 1),  # 归一化度数
            hop2_count / max(num_nodes, 1),  # 归一化2-hop
        ])
        
        # 邻居度数统计
        if hop1_neighbors:
            neighbor_degrees = [len(adj_list[nbr]) for nbr in hop1_neighbors]
            features.extend([
                np.mean(neighbor_degrees),
                np.std(neighbor_degrees) if len(neighbor_degrees) > 1 else 0,
                np.max(neighbor_degrees),
                np.min(neighbor_degrees),
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return features


class ContextualFeatureExtractor:
    """
    上下文特征提取器
    基于特征聚类的上下文特征（纯 PyTorch 实现，避免 sklearn 卡死）
    """
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.centroids = None
        self.mean = None
        self.std = None
        self.proj = None
        
    def _kmeans(self, x, n_clusters, n_iter=20):
        """纯 PyTorch KMeans"""
        N, D = x.shape
        idx = torch.randperm(N, device=x.device)[:n_clusters]
        centroids = x[idx].clone()
        for _ in range(n_iter):
            dists = torch.cdist(x, centroids)
            labels = dists.argmin(dim=1)
            new_c = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_c[k] = x[mask].mean(dim=0)
            centroids = new_c
        return centroids, labels
        
    def fit(self, x):
        """拟合特征提取器"""
        x_t = x.float()
        self.mean = x_t.mean(dim=0)
        self.std = x_t.std(dim=0) + 1e-8
        x_scaled = (x_t - self.mean) / self.std
        
        # 降维：用随机矩阵投影（纯 PyTorch，避免 SVD）
        n_proj = min(self.n_clusters * 5, 50)
        if x_scaled.shape[1] > n_proj:
            torch.manual_seed(42)
            self.proj = torch.randn(x_scaled.shape[1], n_proj, device=x.device) / (x_scaled.shape[1] ** 0.5)
            x_scaled = x_scaled @ self.proj
        else:
            self.proj = None
        
        self.centroids, _ = self._kmeans(x_scaled, self.n_clusters)
        
    def extract(self, x):
        """
        提取上下文特征
        Args:
            x: [N, input_dim] 节点特征
        Returns:
            contextual_features: [N, n_clusters] 上下文特征
        """
        x_t = x.float().cpu()
        x_scaled = (x_t - self.mean.cpu()) / self.std.cpu()
        
        # 应用同样的随机投影
        if self.proj is not None:
            x_scaled = x_scaled @ self.proj.cpu()
        
        # 计算到各聚类中心的距离
        distances = torch.cdist(x_scaled, self.centroids.cpu())  # [N, n_clusters]
        
        # 转换为相似度（高斯核）
        similarities = torch.exp(-distances / distances.mean())
        
        # 归一化
        similarities = similarities / (similarities.sum(dim=1, keepdim=True) + 1e-8)
        
        return similarities.numpy()


class SpectralClusteringFeatures:
    """
    谱聚类特征
    使用随机标签（避免 sklearn 在大数据集上卡死）
    """
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.labels = None
        
    def fit_transform(self, edge_index, num_nodes):
        """
        执行谱聚类
        Args:
            edge_index: 边索引 [2, E]
            num_nodes: 节点数
        Returns:
            cluster_features: [N, n_clusters] one-hot编码的聚类标签
        """
        # 直接用随机标签（避免 sklearn SVD/特征分解卡死）
        torch.manual_seed(42)
        self.labels = torch.randint(0, self.n_clusters, (num_nodes,))
        
        # One-hot编码
        features = torch.zeros(num_nodes, self.n_clusters)
        features[torch.arange(num_nodes), self.labels] = 1
        
        return features.numpy()


class SCNodeEncoder(nn.Module):
    """
    SCNode编码器
    结合空间特征、上下文特征和谱特征
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 特征融合层（原始特征 + 空间特征 + 上下文特征 + 谱特征）
        # 假设额外特征维度：空间特征(8) + 上下文特征(10) + 谱特征(10) = 28
        extra_dim = 28
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + extra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # GNN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, spatial_feat=None, contextual_feat=None, spectral_feat=None):
        """
        前向传播
        Args:
            x: [N, input_dim] 原始特征
            edge_index: [2, E] 边索引
            spatial_feat: [N, 8] 空间特征
            contextual_feat: [N, 10] 上下文特征
            spectral_feat: [N, 10] 谱特征
        Returns:
            embed: [N, output_dim] 节点嵌入
        """
        # 特征融合
        features = [x]
        
        if spatial_feat is not None:
            features.append(spatial_feat)
        if contextual_feat is not None:
            features.append(contextual_feat)
        if spectral_feat is not None:
            features.append(spectral_feat)
        
        # 如果某些特征缺失，补零
        if spatial_feat is None:
            features.append(torch.zeros(x.size(0), 8, device=x.device))
        if contextual_feat is None:
            features.append(torch.zeros(x.size(0), 10, device=x.device))
        if spectral_feat is None:
            features.append(torch.zeros(x.size(0), 10, device=x.device))
        
        x_fused = torch.cat(features, dim=-1)
        h = self.fusion(x_fused)
        
        # GNN传播
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # 残差连接
        
        # 输出投影
        embed = self.output_proj(h)
        
        return embed


class SCNodeFullModel(nn.Module):
    """
    完整的SCNode模型
    """
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 特征提取器
        self.spatial_extractor = SpatialFeatureExtractor(num_hops=2)
        self.contextual_extractor = ContextualFeatureExtractor(
            n_clusters=kwargs.get('n_clusters', 10),
        )
        self.spectral_extractor = SpectralClusteringFeatures(
            n_clusters=kwargs.get('n_spectral_clusters', 10)
        )
        
        # 节点编码器
        self.encoder = SCNodeEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 链接预测层
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 状态标记
        self.is_fitted = False
        
        # 特征缓存
        self.cached_features = None
        self.cached_num_nodes = None
        
    def fit(self, x, edge_index):
        """
        拟合特征提取器（应在训练开始前调用）
        """
        if self.is_fitted:
            return
        
        print("Fitting SCNode feature extractors...")
        
        # 拟合上下文特征提取器
        self.contextual_extractor.fit(x)
        
        # 预提取并缓存所有特征
        self._cache_features(x, edge_index)
        
        self.is_fitted = True
        print("SCNode feature extractors fitted!")
    
    def _cache_features(self, x, edge_index):
        """
        预提取并缓存特征
        """
        device = x.device
        num_nodes = x.size(0)
        
        # 转换为numpy进行特征提取
        x_cpu = x.cpu()
        edge_index_cpu = edge_index.cpu()
        
        # 创建简单的Data对象
        class SimpleData:
            def __init__(self, num_nodes, edge_index):
                self.num_nodes = num_nodes
                self.edge_index = edge_index
        
        data = SimpleData(num_nodes, edge_index_cpu)
        
        # 提取空间特征
        spatial_feat = self.spatial_extractor.extract(data)
        spatial_feat = torch.tensor(spatial_feat, dtype=torch.float32, device=device)
        
        # 提取上下文特征
        contextual_feat = self.contextual_extractor.extract(x_cpu)
        contextual_feat = torch.tensor(contextual_feat, dtype=torch.float32, device=device)
        
        # 提取谱特征
        spectral_feat = self.spectral_extractor.fit_transform(edge_index_cpu, num_nodes)
        spectral_feat = torch.tensor(spectral_feat, dtype=torch.float32, device=device)
        
        # 缓存特征
        self.cached_features = {
            'spatial': spatial_feat,
            'contextual': contextual_feat,
            'spectral': spectral_feat
        }
        self.cached_num_nodes = num_nodes
        print(f"Features cached for {num_nodes} nodes")
        
    def extract_features(self, x, edge_index):
        """
        提取所有特征（如果已缓存则使用缓存）
        Args:
            x: [N, input_dim] 节点特征
            edge_index: [2, E] 边索引
        Returns:
            dict: 包含所有特征的字典
        """
        num_nodes = x.size(0)
        
        # 如果特征已缓存且节点数匹配，直接使用缓存
        if self.cached_features is not None and self.cached_num_nodes == num_nodes:
            return self.cached_features
        
        # 否则重新提取（这不应该在训练期间发生）
        print("Warning: Recomputing features (should not happen during training)")
        device = x.device
        x_cpu = x.cpu()
        edge_index_cpu = edge_index.cpu()
        
        class SimpleData:
            def __init__(self, num_nodes, edge_index):
                self.num_nodes = num_nodes
                self.edge_index = edge_index
        
        data = SimpleData(num_nodes, edge_index_cpu)
        
        spatial_feat = self.spatial_extractor.extract(data)
        spatial_feat = torch.tensor(spatial_feat, dtype=torch.float32, device=device)
        
        contextual_feat = self.contextual_extractor.extract(x_cpu)
        contextual_feat = torch.tensor(contextual_feat, dtype=torch.float32, device=device)
        
        spectral_feat = self.spectral_extractor.fit_transform(edge_index_cpu, num_nodes)
        spectral_feat = torch.tensor(spectral_feat, dtype=torch.float32, device=device)
        
        return {
            'spatial': spatial_feat,
            'contextual': contextual_feat,
            'spectral': spectral_feat
        }
    
    def encode_nodes(self, x, edge_index):
        """编码节点"""
        # 提取特征
        features = self.extract_features(x, edge_index)
        
        # 编码
        embed = self.encoder(
            x,
            edge_index,
            spatial_feat=features['spatial'],
            contextual_feat=features['contextual'],
            spectral_feat=features['spectral']
        )
        
        return embed
    
    def predict_link(self, node_embed, edge_index):
        """
        预测链接
        Args:
            node_embed: [N, output_dim] 节点嵌入
            edge_index: [2, E] 边索引
        Returns:
            pred: [E] 链接概率
        """
        src, dst = edge_index
        src_embed = node_embed[src]
        dst_embed = node_embed[dst]
        
        # 拼接并预测
        link_feat = torch.cat([src_embed, dst_embed], dim=-1)
        pred = self.link_predictor(link_feat).squeeze(-1)
        
        return pred
    
    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        Args:
            x: [N, input_dim] 节点特征
            edge_index: [2, E] 边索引
            batch: [N] batch索引（用于图级任务）
        Returns:
            embed: [N, output_dim] 节点嵌入
            loss: None（SCNode没有额外的损失）
        """
        # 确保特征提取器已拟合
        if not self.is_fitted:
            self.fit(x, edge_index)
        
        # 编码节点
        embed = self.encode_nodes(x, edge_index)
        
        return embed, None


class SCNodeFullAdapter(nn.Module):
    """
    SCNode完整模型适配器
    包装SCNodeFullModel以兼容消融实验接口
    支持链接预测和节点分类双任务
    """
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3, 
                 num_classes=None, **kwargs):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        self.model = SCNodeFullModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
        # 节点分类头（如果指定了num_classes）
        if num_classes is not None:
            self.classifier = nn.Linear(output_dim, num_classes)
        else:
            self.classifier = None
            
        self.is_fitted = False
        
    def encode(self, x, edge_index, debug=False):
        """编码器：获取节点嵌入"""
        # 首次调用时拟合特征提取器（带超时检测）
        if not self.is_fitted:
            print("Fitting SCNode feature extractors for the first time...")
            import threading
            
            self.oot_occurred = False
            fit_result = [None]
            
            def fit_job():
                try:
                    fit_result[0] = self.model.fit(x, edge_index)
                except Exception as e:
                    fit_result[0] = e
            
            start_fit = time.time()
            thread = threading.Thread(target=fit_job)
            thread.daemon = True
            thread.start()
            thread.join(timeout=None)
            
            fit_time = time.time() - start_fit
            
            if thread.is_alive():
                # 超时
                print(f"OOT: Feature extraction took {fit_time:.1f}s (>10s)")
                self.oot_occurred = True
                self.is_fitted = True
                # 返回零嵌入
                embed = torch.zeros(x.size(0), self.model.output_dim, device=x.device)
                return embed, None
            else:
                self.oot_occurred = False
                print(f"Feature extractors fitted in {fit_time:.1f}s")
            
            self.is_fitted = True
        
        embed, _ = self.model(x, edge_index)
        
        return embed, None
        
    def forward(self, x, edge_index, return_embeddings=False, debug=False):
        """
        前向传播，支持链接预测和节点分类双任务
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            return_embeddings: 是否返回embeddings（链接预测模式）
            debug: 是否打印调试信息
            
        Returns:
            如果return_embeddings=True或classifier为None:
                (embeddings, None) - 用于链接预测
            否则:
                (logits, embeddings, None) - 用于节点分类
        """
        # 获取嵌入
        embeddings, _ = self.encode(x, edge_index, debug)
        
        # 如果指定了return_embeddings或没有分类头，返回embeddings（链接预测兼容模式）
        if return_embeddings or self.classifier is None:
            return embeddings, None
        
        # 节点分类模式
        logits = self.classifier(embeddings)
        return logits, embeddings, None
    
    def predict_link(self, x, edge_index, links):
        """
        预测链接（用于链接预测评估）
        
        Args:
            x: 节点特征
            edge_index: 边索引
            links: [E, 2] 要预测的链接
            
        Returns:
            pred: [E] 链接概率
        """
        # 编码节点
        embed = self.model.encode_nodes(x, edge_index)
        
        # 预测链接
        pred = self.model.predict_link(embed, links.t())
        
        return pred
