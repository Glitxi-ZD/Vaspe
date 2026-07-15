"""
完整的 GPM (Graph Pattern Mining) 模型实现
基于原始论文代码，适配链接预测任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import hashlib
from torch_geometric.nn import Node2Vec
from torch_scatter import scatter_add
from torch_cluster import random_walk


class VectorQuantize(nn.Module):
    """
    向量量化模块 (VQ-VAE)
    用于学习离散的图模式表示
    """
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=True,
        use_cosine_sim=True,
        kmeans_init=True,
        ema_update=True,
        decay=0.99,
        eps=1e-5
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim if codebook_dim is not None else dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head
        self.use_cosine_sim = use_cosine_sim
        self.kmeans_init = kmeans_init
        self.ema_update = ema_update
        self.decay = decay
        self.eps = eps
        
        # 确定codebook的形状
        if separate_codebook_per_head:
            self.codebook_shape = (heads, codebook_size, self.codebook_dim // heads)
        else:
            self.codebook_shape = (1, codebook_size, self.codebook_dim)
        
        # 初始化codebook
        self.codebook = nn.Parameter(torch.randn(*self.codebook_shape))
        
        # EMA更新参数
        if ema_update:
            self.register_buffer('ema_cluster_size', torch.zeros(*self.codebook_shape[:2]))
            self.register_buffer('ema_w', self.codebook.data.clone())
        
        # 投影层
        self.proj_in = nn.Linear(dim, self.codebook_dim)
        self.proj_out = nn.Linear(self.codebook_dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: [..., dim] 输入特征
        Returns:
            quantized: [..., dim] 量化后的特征
            indices: [...] codebook索引
            commit_loss: 标量 commitment loss
            perplexity: 标量 codebook使用率
        """
        # 投影到codebook维度
        x = self.proj_in(x)
        
        # 分割多head
        if self.heads > 1:
            x = x.reshape(*x.shape[:-1], self.heads, -1)
            x = x.transpose(-2, -3)  # [..., heads, dim//heads]
        else:
            x = x.unsqueeze(-3)  # [..., 1, dim]
        
        # 计算与codebook的距离
        if self.use_cosine_sim:
            # 余弦相似度
            x_norm = F.normalize(x, dim=-1)
            codebook_norm = F.normalize(self.codebook, dim=-1)
            dist = 1 - (x_norm @ codebook_norm.transpose(-2, -1))  # [..., heads, codebook_size]
        else:
            # 欧氏距离
            dist = torch.sum((x.unsqueeze(-2) - self.codebook) ** 2, dim=-1)
        
        # 找到最近的codebook向量
        indices = torch.argmin(dist, dim=-1)  # [..., heads]
        
        # 量化
        encodings = F.one_hot(indices, self.codebook_size).float()
        quantized = encodings @ self.codebook  # [..., heads, codebook_dim//heads]
        
        # 恢复形状
        if self.heads > 1:
            quantized = quantized.transpose(-2, -3)
            quantized = quantized.reshape(*quantized.shape[:-2], -1)
        else:
            quantized = quantized.squeeze(-3)
        
        # EMA更新
        if self.training and self.ema_update:
            with torch.no_grad():
                # 计算每个codebook向量的使用次数
                encodings_flat = encodings.reshape(-1, self.codebook_size)
                cluster_size = encodings_flat.sum(dim=0)
                
                # 更新EMA聚类大小
                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size, alpha=1 - self.decay
                )
                
                # 更新EMA codebook
                x_flat = x.reshape(-1, x.shape[-1])
                emb_sum = encodings_flat.t() @ x_flat
                self.ema_w.mul_(self.decay).add_(emb_sum, alpha=1 - self.decay)
                
                # 更新codebook
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                )
                self.codebook.data = self.ema_w / cluster_size.unsqueeze(-1)
        
        # Commitment loss（straight-through estimator）
        commit_loss = F.mse_loss(x.reshape_as(quantized), quantized.detach())
        
        # Straight-through estimator
        quantized = x.reshape_as(quantized) + (quantized - x.reshape_as(quantized)).detach()
        
        # 计算perplexity（codebook使用率）
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 投影回原始维度
        quantized = self.proj_out(quantized)
        
        return quantized, indices, commit_loss, perplexity


class PatternEncoder(nn.Module):
    """
    模式编码器
    编码图模式（k-hop邻居子图）
    """
    def __init__(self, params):
        super().__init__()
        self.input_dim = params.get('input_dim', 384)
        self.hidden_dim = params.get('hidden_dim', 64)
        self.node_pe_dim = params.get('node_pe_dim', 0)
        self.edge_dim = params.get('edge_dim', 0)
        
        # 特征投影
        total_input_dim = self.input_dim + self.node_pe_dim
        if self.edge_dim > 0:
            total_input_dim += self.edge_dim
            
        self.proj = nn.Sequential(
            nn.Linear(total_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(params.get('dropout', 0.1))
        )
        
    def encode_node(self, patterns, feat, node_pe, e_feat, params):
        """
        编码节点模式
        Args:
            patterns: [num_patterns, num_nodes, k] 模式（k-hop邻居索引）
            feat: [N, input_dim] 节点特征
            node_pe: [N, pe_dim] 位置编码
            e_feat: [E, edge_dim] 边特征（可选）
        Returns:
            pattern_feat: [num_patterns, num_nodes, hidden_dim]
        """
        num_patterns, num_nodes, k = patterns.shape
        
        # 收集模式中的节点特征
        pattern_nodes = patterns.reshape(-1)  # [num_patterns * num_nodes * k]
        pattern_feat = feat[pattern_nodes]  # [num_patterns * num_nodes * k, input_dim]
        
        # 添加位置编码
        if node_pe is not None:
            pe_feat = node_pe[pattern_nodes]
            pattern_feat = torch.cat([pattern_feat, pe_feat], dim=-1)
        
        # 重塑为 [num_patterns, num_nodes, k, feat_dim]
        feat_dim = pattern_feat.shape[-1]
        pattern_feat = pattern_feat.reshape(num_patterns, num_nodes, k, feat_dim)
        
        # 投影
        pattern_feat = self.proj(pattern_feat)  # [num_patterns, num_nodes, k, hidden_dim]
        
        # 平均池化
        pattern_feat = pattern_feat.mean(dim=2)  # [num_patterns, num_nodes, hidden_dim]
        
        return pattern_feat


class LinkPredictor(nn.Module):
    """链接预测头"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class Consistency(nn.Module):
    """一致性损失（用于自监督）"""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, probs):
        avg_probs = probs.mean(0)
        sharpened_probs = avg_probs.pow(1 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdim=True)
        loss = (sharpened_probs - avg_probs).pow(2).sum(-1).mean()
        return loss


class GPMFullModel(nn.Module):
    """
    完整的GPM模型
    添加预计算模式缓存机制
    """
    def __init__(self, params):
        super().__init__()
        
        # 参数设置
        self.input_dim = params.get('input_dim', 384)
        self.hidden_dim = params.get('hidden_dim', 64)
        self.output_dim = params.get('output_dim', 64)
        self.num_layers = params.get('num_layers', 2)
        self.codebook_size = params.get('codebook_size', 512)
        self.heads = params.get('heads', 4)
        self.use_vq = params.get('use_vq', True)
        self.use_cls_token = params.get('use_cls_token', True)
        self.use_attn_fusion = params.get('use_attn_fusion', True)
        self.dropout = params.get('dropout', 0.1)
        self.norm_first = params.get('norm_first', True)
        self.k = params.get('k', 2)  # k-hop 邻居数
        self.num_patterns = params.get('num_patterns', 5)  # 模式数量
        
        # 缓存设置
        self.cache_dir = params.get('cache_dir', './.gpm_cache')
        self.use_cache = params.get('use_cache', True)
        self._pattern_cache = {}  # 内存缓存
        
        # 输入投影
        total_input = self.input_dim + params.get('node_pe_dim', 0) + params.get('edge_dim', 0)
        self.input_proj = nn.Linear(total_input, self.hidden_dim)
        
        # 模式编码器
        self.pattern_encoder = PatternEncoder(params)
        
        # 向量量化
        if self.use_vq:
            self.vq = VectorQuantize(
                dim=self.hidden_dim,
                codebook_size=self.codebook_size,
                codebook_dim=self.hidden_dim,
                heads=self.heads,
                separate_codebook_per_head=True,
                use_cosine_sim=True,
                kmeans_init=True,
                ema_update=True,
            )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            norm_first=self.norm_first,
            batch_first=False
        )
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                norm_first=self.norm_first,
                batch_first=False
            ) for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        # 链接预测头
        self.head = LinkPredictor(self.hidden_dim, self.hidden_dim, 1, 3, 0.0)
        
        # 注意力融合
        if self.use_attn_fusion:
            attn_dim = self.hidden_dim if not self.use_cls_token else 2 * self.hidden_dim
            self.attn_layer = nn.Linear(attn_dim, 1)
        
        # 一致性损失
        self.consistency = Consistency(temperature=0.5)
        
    def transformer_encode(self, x):
        """Transformer编码"""
        for layer in self.encoder:
            last_x = x
            x = self.norm(x)
            x = layer(x)
            x = last_x + x
        return x
    
    def get_instance_emb(self, pattern_emb):
        """从模式嵌入获取实例嵌入"""
        if self.use_cls_token:
            if self.use_attn_fusion:
                target = pattern_emb[0]  # [num_nodes, hidden_dim]
                source = pattern_emb[1:]  # [num_patterns-1, num_nodes, hidden_dim]
                
                # 计算注意力权重
                attn_input = torch.cat([
                    target.unsqueeze(0).expand(source.size(0), -1, -1),
                    source
                ], dim=-1)  # [num_patterns-1, num_nodes, 2*hidden_dim]
                attn = self.attn_layer(attn_input).squeeze(-1)  # [num_patterns-1, num_nodes]
                attn = F.softmax(attn, dim=0)
                
                # 加权求和
                instance_emb = torch.sum(attn.unsqueeze(-1) * source, dim=0) + target
            else:
                instance_emb = pattern_emb[0]
        else:
            if self.use_attn_fusion:
                attn = self.attn_layer(pattern_emb).squeeze(-1)  # [num_patterns, num_nodes]
                attn = F.softmax(attn, dim=0)
                instance_emb = torch.sum(attn.unsqueeze(-1) * pattern_emb, dim=0)
            else:
                instance_emb = pattern_emb.mean(dim=0)
        
        return instance_emb
    
    def encode_link(self, graph, links, mode='train', batch_size=256):
        """
        编码链接
        Args:
            graph: 图数据对象
            links: [E, 2] 链接索引
            mode: 'train' 或 'eval'
            batch_size: batch大小，用于控制显存使用
        """
        device = next(self.parameters()).device
        feat = graph.x
        node_pe = getattr(graph, 'pe', None)
        
        # 分batch处理链接以避免OOM
        num_links = links.size(0)
        all_preds = []
        all_embs = []
        all_commit_losses = []
        
        for start_idx in range(0, num_links, batch_size):
            end_idx = min(start_idx + batch_size, num_links)
            batch_links = links[start_idx:end_idx]
            
            source_nodes, target_nodes = batch_links[:, 0], batch_links[:, 1]
            all_nodes = {'source': source_nodes, 'target': target_nodes}
            
            instance_embs = {}
            commit_losses = []
            
            # 对源节点和目标节点分别编码
            for key, nodes in all_nodes.items():
                # 提取k-hop邻居作为模式（使用缓存）
                patterns = self.extract_patterns(graph, nodes, k=self.k, num_patterns=self.num_patterns)
                
                # 编码模式
                pattern_feat = self.pattern_encoder.encode_node(patterns, feat, node_pe, None, {})
                
                # 向量量化
                if self.use_vq:
                    pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
                    commit_losses.append(commit_loss)
                
                # 添加CLS token
                if self.use_cls_token:
                    cls_token = torch.ones(1, pattern_feat.size(1), pattern_feat.size(2), device=device)
                    pattern_feat = torch.cat([cls_token, pattern_feat], dim=0)
                
                # Transformer编码
                pattern_emb = self.transformer_encode(pattern_feat)
                
                # 获取实例嵌入
                instance_emb = self.get_instance_emb(pattern_emb)
                instance_embs[key] = instance_emb
            
            # 组合源节点和目标节点的嵌入
            combined_emb = instance_embs['source'] * instance_embs['target']
            
            # 链接预测
            pred = self.head(combined_emb)
            
            all_preds.append(pred.squeeze(-1))
            all_embs.append(combined_emb)
            all_commit_losses.extend(commit_losses)
        
        # 合并所有batch结果
        total_pred = torch.cat(all_preds, dim=0)
        total_emb = torch.cat(all_embs, dim=0)
        total_commit_loss = sum(all_commit_losses) if all_commit_losses else 0
        
        return total_pred, total_emb, total_commit_loss
    
    def _get_cache_key(self, edge_index, num_nodes):
        """生成缓存键"""
        # 使用 CPU tensor 避免 CUDA OOM
        edge_index_cpu = edge_index.cpu()
        edge_str = f"{edge_index_cpu[0].sum().item()}_{edge_index_cpu[1].sum().item()}_{num_nodes}"
        return hashlib.md5(edge_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key):
        """获取缓存文件路径"""
        cache_file = f"patterns_{cache_key}_{self.num_patterns}_{self.k}.pt"
        return os.path.join(self.cache_dir, cache_file)
    
    def extract_patterns(self, graph, nodes, k=2, num_patterns=5):
        """
        提取节点的k-hop邻居模式
        使用 torch_cluster.random_walk 进行高效模式提取
        支持内存缓存和磁盘缓存
        """
        device = next(self.parameters()).device
        edge_index = graph.edge_index
        num_nodes_total = nodes.size(0)
        
        # 尝试从内存缓存获取
        cache_key = self._get_cache_key(edge_index, num_nodes_total)
        if self.use_cache and cache_key in self._pattern_cache:
            cached_patterns = self._pattern_cache[cache_key]
            # 验证缓存形状是否正确
            if cached_patterns.shape[1] == num_nodes_total:
                return cached_patterns
            else:
                # 缓存不匹配，重新计算
                del self._pattern_cache[cache_key]
        
        # 尝试从磁盘缓存获取
        cache_path = self._get_cache_path(cache_key)
        if self.use_cache and os.path.exists(cache_path):
            try:
                patterns = torch.load(cache_path, map_location=device)
                # 验证缓存形状是否正确
                if patterns.shape[1] == num_nodes_total:
                    self._pattern_cache[cache_key] = patterns
                    return patterns
                else:
                    # 缓存不匹配，删除并重新计算
                    os.remove(cache_path)
                    print(f"[WARNING] Cache shape mismatch, regenerating patterns for {cache_key}")
            except Exception as e:
                print(f"[WARNING] Failed to load cache, regenerating patterns: {e}")
                # 删除损坏的缓存文件
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        # 使用 random_walk 提取模式
        row, col = edge_index
        nodes_cpu = nodes.cpu()
        
        all_patterns = []
        batch_size = 1024  # 批处理大小
        
        for start_idx in range(0, num_nodes_total, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes_total)
            batch_nodes = nodes[start_idx:end_idx]
            batch_size_actual = batch_nodes.size(0)
            
            # 重复节点以匹配 num_patterns
            batch_nodes_repeated = batch_nodes.repeat(num_patterns)
            
            # 执行 random_walk
            walk_length = k  # 步数
            try:
                patterns_batch, _ = random_walk(row, col, start=batch_nodes_repeated, 
                                               walk_length=walk_length, return_edge_indices=True)
            except Exception as e:
                print(f"[ERROR] random_walk failed: {e}")
                # 如果失败，返回空模式
                patterns_batch = torch.zeros(num_patterns * batch_size_actual, walk_length + 1, dtype=torch.long, device=row.device)
            
            # reshape: [num_patterns * batch_size, k+1] -> [num_patterns, batch_size, k+1]
            patterns_batch = patterns_batch.view(num_patterns, batch_size_actual, -1)
            all_patterns.append(patterns_batch)
        
        # 合并所有批次
        patterns = torch.cat(all_patterns, dim=1)  # [num_patterns, num_nodes, k+1]
        patterns = patterns.to(device)
        
        # 验证patterns的索引范围
        max_node_id = edge_index.max().item() if edge_index.numel() > 0 else 0
        if patterns.max() > max_node_id:
            print(f"[WARNING] Pattern indices exceed node count, clamping to valid range")
            patterns = torch.clamp(patterns, min=0, max=max_node_id)
        
        # 保存到缓存
        if self.use_cache:
            self._pattern_cache[cache_key] = patterns
            os.makedirs(self.cache_dir, exist_ok=True)
            try:
                torch.save(patterns, cache_path)
            except Exception as e:
                print(f"[WARNING] Failed to save cache: {e}")
        
        return patterns
    
    def _get_khop_neighbors(self, edge_index, node, k):
        """BFS获取k-hop邻居（备用方法）"""
        neighbors = [node]
        visited = {node}
        
        for _ in range(k):
            new_neighbors = []
            for n in neighbors:
                # 查找邻居
                mask = (edge_index[0] == n) | (edge_index[1] == n)
                nbrs = edge_index[:, mask].flatten().unique().tolist()
                for nbr in nbrs:
                    if nbr not in visited:
                        visited.add(nbr)
                        new_neighbors.append(nbr)
            neighbors = new_neighbors
            if not neighbors:
                break
        
        return list(visited)
    
    def forward(self, x, edge_index, batch=None, mode='train', batch_size=256):
        """
        标准前向传播（兼容链接预测）
        Args:
            x: 节点特征
            edge_index: 边索引
            batch: batch信息
            mode: 'train' 或 'eval'
            batch_size: batch大小，用于控制显存使用
        """
        # 创建简单的图对象
        class SimpleGraph:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index
                self.pe = None
        
        graph = SimpleGraph(x, edge_index)
        
        # 生成训练/测试链接
        num_nodes = x.size(0)
        if mode == 'train':
            # 使用现有边作为正样本
            links = edge_index.t()
        else:
            # 生成候选链接
            links = self._sample_candidate_links(edge_index, num_nodes)
        
        pred, emb, commit_loss = self.encode_link(graph, links, mode, batch_size=batch_size)
        
        return emb, commit_loss
    
    def _sample_candidate_links(self, edge_index, num_nodes, num_samples=1000):
        """采样候选链接用于评估"""
        device = edge_index.device
        # 随机采样
        source = torch.randint(0, num_nodes, (num_samples,), device=device)
        target = torch.randint(0, num_nodes, (num_samples,), device=device)
        links = torch.stack([source, target], dim=1)
        return links


class GPMFullAdapter(nn.Module):
    """
    GPM完整模型适配器
    包装GPMFullModel以兼容消融实验接口
    """
    def __init__(self, input_dim=384, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.3, 
                 cache_dir='./.gpm_cache', use_cache=True, batch_size=64, **kwargs):
        super().__init__()
        
        # GPM参数 - 使用更小的默认值以节省显存
        params = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'codebook_size': kwargs.get('codebook_size', 128),  # 从512减少到128
            'heads': kwargs.get('heads', 2),  # 从4减少到2
            'use_vq': kwargs.get('use_vq', True),
            'use_cls_token': kwargs.get('use_cls_token', False),  # 默认关闭以节省显存
            'use_attn_fusion': kwargs.get('use_attn_fusion', False),  # 默认关闭以节省显存
            'norm_first': kwargs.get('norm_first', True),
            'node_pe_dim': kwargs.get('node_pe_dim', 0),
            'edge_dim': kwargs.get('edge_dim', 0),
            'cache_dir': cache_dir,
            'use_cache': use_cache,
            'k': kwargs.get('k', 2),
            'num_patterns': kwargs.get('num_patterns', 3),  # 从5减少到3
        }
        
        self.model = GPMFullModel(params)
        self.output_dim = output_dim
        self.batch_size = batch_size
        
    def forward(self, x, edge_index, debug=False):
        """
        前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            debug: 是否打印调试信息
            
        Returns:
            embed: 节点嵌入 [N, output_dim]
            loss: GMM损失（对于GPM是VQ损失）
        """
        embed, vq_loss = self.model(x, edge_index, mode='train', batch_size=self.batch_size)
        
        if vq_loss is None:
            vq_loss = torch.tensor(0.0, device=x.device)
        
        # 返回损失字典以兼容训练代码
        loss_dict = {'vq_loss': vq_loss} if vq_loss is not None else None
        
        return embed, loss_dict
    
    def encode_link(self, x, edge_index, links):
        """编码链接（用于链接预测评估）"""
        class SimpleGraph:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index
                self.pe = None
        
        graph = SimpleGraph(x, edge_index)
        pred, emb, _ = self.model.encode_link(graph, links, mode='eval')
        return pred, emb
