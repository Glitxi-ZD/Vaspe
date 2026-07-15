"""
Baseline 模型包
包含 GCN、GraphSAGE、GAT、GPM、SCNode 等基线模型
"""

from .gcn import GCNBaseline, GCNConv
from .graphsage import GraphSAGEBaseline, SAGEConv
from .gat import GATBaseline, GATConv
from .gpm_simple import GPMSimple, GPMSimpleAdapter

# 可选依赖（需要 torch_geometric 等）
try:
    from .gpm_full import GPMFullModel, GPMFullAdapter
except ImportError:
    GPMFullModel = None
    GPMFullAdapter = None

try:
    from .scnode_full import SCNodeFullModel, SCNodeFullAdapter
except ImportError:
    SCNodeFullModel = None
    SCNodeFullAdapter = None

# 模型注册表
BASELINE_MODELS = {
    'gcn': GCNBaseline,
    'graphsage': GraphSAGEBaseline,
    'gat': GATBaseline,
    'gpm_simple': GPMSimpleAdapter,
}

# 添加可选模型
if GPMFullAdapter is not None:
    BASELINE_MODELS['gpm_full'] = GPMFullAdapter
if SCNodeFullAdapter is not None:
    BASELINE_MODELS['scnode_full'] = SCNodeFullAdapter


def create_baseline_model(model_name, input_dim=384, hidden_dim=64, output_dim=64, 
                          num_classes=None, **kwargs):
    """
    创建 baseline 模型
    
    Args:
        model_name: 模型名称 ('gcn', 'graphsage', 'gpm_simple', 'gpm_full', 'scnode_full')
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_classes: 节点分类类别数（None表示链接预测）
        **kwargs: 其他参数
    
    Returns:
        模型实例
    """
    if model_name not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline model: {model_name}. "
                        f"Available: {list(BASELINE_MODELS.keys())}")
    
    model_class = BASELINE_MODELS[model_name]
    
    # 过滤参数
    valid_params = {
        'gcn': ['input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout', 'num_classes'],
        'graphsage': ['input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout', 'num_classes'],
        'gat': ['input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout', 'num_classes', 'heads'],
        'gpm_simple': ['input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout', 'num_classes', 'codebook_size'],
        'gpm_full': ['input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout', 'num_classes'],
        'scnode_full': ['input_dim', 'hidden_dim', 'output_dim', 'num_layers', 'dropout', 'num_classes'],
    }
    
    if model_name in valid_params:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params[model_name]}
    else:
        filtered_kwargs = kwargs
    
    # 创建模型
    if num_classes is not None:
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            **filtered_kwargs
        )
    else:
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **filtered_kwargs
        )
