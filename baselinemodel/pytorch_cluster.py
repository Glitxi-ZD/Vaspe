"""
PyTorch 实现的 KMeans 和谱聚类
替代 sklearn 的实现，解决 Windows 上的 DLL 兼容性问题
"""

import torch
import numpy as np


class PyTorchKMeans:
    """
    纯 PyTorch 实现的 KMeans 聚类
    """
    def __init__(self, n_clusters=10, max_iter=50, tol=1e-4, random_state=42, n_init=3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, X):
        """
        拟合 KMeans 模型
        Args:
            X: numpy array or torch tensor, shape [N, D]
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to('cpu')
        
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        
        torch.manual_seed(self.random_state)
        
        for init_idx in range(self.n_init):
            # 初始化聚类中心（随机选择）
            perm = torch.randperm(X.size(0))
            centers = X[perm[:self.n_clusters]].clone()
            
            for iteration in range(self.max_iter):
                # 分配样本到最近的聚类中心
                distances = self._compute_distances(X, centers)
                labels = distances.argmin(dim=1)
                
                # 更新聚类中心
                new_centers = torch.zeros_like(centers)
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if mask.sum() > 0:
                        new_centers[k] = X[mask].mean(dim=0)
                    else:
                        new_centers[k] = X[torch.randint(0, X.size(0), (1,))]
                
                # 检查收敛
                center_shift = (new_centers - centers).norm(dim=1).max().item()
                centers = new_centers
                
                if center_shift < self.tol:
                    break
            
            # 计算惯性（inertia）
            distances = self._compute_distances(X, centers)
            labels = distances.argmin(dim=1)
            inertia = distances.gather(1, labels.unsqueeze(1)).sum().item()
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.clone()
                best_labels = labels.clone()
        
        self.cluster_centers_ = best_centers.numpy()
        self.labels_ = best_labels.numpy()
        
        return self
    
    def predict(self, X):
        """预测新样本的聚类标签"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        centers = torch.from_numpy(self.cluster_centers_).float()
        distances = self._compute_distances(X, centers)
        return distances.argmin(dim=1).numpy()
    
    def transform(self, X):
        """计算到各聚类中心的距离"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        centers = torch.from_numpy(self.cluster_centers_).float()
        distances = self._compute_distances(X, centers)
        return distances.numpy()
    
    def _kmeans_plus_plus_init(self, X):
        """K-Means++ 初始化"""
        n_samples = X.size(0)
        centers = []
        
        # 随机选择第一个中心
        idx = torch.randint(0, n_samples, (1,)).item()
        centers.append(X[idx])
        
        for _ in range(1, self.n_clusters):
            centers_tensor = torch.stack(centers)
            distances = self._compute_distances(X, centers_tensor)
            min_distances = distances.min(dim=1)[0]
            
            # 按距离概率选择下一个中心
            probs = min_distances / min_distances.sum()
            idx = torch.multinomial(probs, 1).item()
            centers.append(X[idx])
        
        return torch.stack(centers)
    
    def _compute_distances(self, X, centers):
        """计算每个样本到每个聚类中心的距离"""
        # X: [N, D], centers: [K, D]
        # 返回: [N, K]
        return torch.cdist(X, centers)


class PyTorchSpectralClustering:
    """
    纯 PyTorch 实现的谱聚类
    """
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        
    def fit_predict(self, edge_index, num_nodes):
        """
        执行谱聚类
        Args:
            edge_index: 边索引 [2, E]
            num_nodes: 节点数
        Returns:
            labels: [N] 聚类标签
        """
        # 构建邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        
        # 计算度矩阵
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 计算归一化拉普拉斯矩阵: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # 计算前 k 个特征向量
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            # 取前 k 个最小的特征值对应的特征向量
            indices = eigenvalues.argsort()[:self.n_clusters]
            U = eigenvectors[:, indices]
            
            # 对行进行归一化
            U_norm = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            
            # 在特征空间中进行 KMeans
            kmeans = PyTorchKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            kmeans.fit(U_norm.numpy())
            self.labels_ = kmeans.labels_
            
        except Exception as e:
            print(f"Spectral clustering failed: {e}, using random initialization")
            self.labels_ = np.random.randint(0, self.n_clusters, size=num_nodes)
        
        return self.labels_
