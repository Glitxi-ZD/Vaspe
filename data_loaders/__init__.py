"""
数据加载器包
包含各种数据集的加载函数
"""

from .cora_data import create_cora_dataset
from .citeseer_data import create_citeseer_dataset
from .pubmed_data import create_pubmed_dataset
from .coauthor_cs_data import create_coauthor_cs_dataset
from .wikics_data import create_wikics_dataset
from .amazon_computers_data import create_amazon_computers_dataset
from .custom_data import create_custom_dataset

# 数据集名称到加载函数的映射
DATASET_LOADERS = {
    'cora': create_cora_dataset,
    'citeseer': create_citeseer_dataset,
    'pubmed': create_pubmed_dataset,
    'coauthor_cs': create_coauthor_cs_dataset,
    'wikics': create_wikics_dataset,
    'amazon_computers': create_amazon_computers_dataset,
    'custom': create_custom_dataset,
}


def load_dataset(dataset_name, task='node_classification', **kwargs):
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称
        task: 任务类型 ('node_classification' 或 'link_prediction')
    
    Returns:
        数据集对象
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_LOADERS.keys())}")
    
    loader = DATASET_LOADERS[dataset_name]
    return loader(task=task, **kwargs)
