# VASPE: Variational Autoencoder with Spectral Filters for Graph Representation Learning

Official implementation of **VASPE** — a graph representation learning framework combining Variational Autoencoders with spectral-domain frequency-selective filters.

## Installation

```bash
git clone https://github.com/your-username/VASPE.git
cd VASPE

conda create -n vaspe python=3.10
conda activate vaspe

pip install torch torchvision torch_geometric
pip install scikit-learn numpy
```

## Usage

### Node Classification

```bash
python run_node_classification.py                          # All 6 datasets
python run_node_classification.py --datasets cora citeseer  # Specific datasets
python run_node_classification.py --device cpu              # CPU mode
```

### Link Prediction

```bash
python run_link_prediction.py                              # All 6 datasets
python run_link_prediction.py --datasets cora pubmed       # Specific datasets
python run_link_prediction.py --device cpu                 # CPU mode
```

Use `--help` to see all available hyperparameter options.

## Datasets

Datasets are automatically downloaded and cached in `data/` on first run:

| Dataset | Nodes | Features | Classes |
|---------|-------|----------|---------|
| Cora | 2,708 | 1,433 | 7 |
| CiteSeer | 3,327 | 3,703 | 6 |
| PubMed | 19,717 | 500 | 3 |
| Coauthor CS | 18,333 | 6,805 | 15 |
| WikiCS | 11,701 | 300 | 10 |
| Amazon Computers | 13,752 | 767 | 10 |

