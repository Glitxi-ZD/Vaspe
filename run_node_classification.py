import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from mymodel import VAEFullModel

CITATION_DATASETS = {'cora', 'citeseer', 'pubmed'}
ALL_DATASETS = ['cora', 'citeseer', 'pubmed', 'coauthor_cs', 'wikics', 'amazon_computers']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CitationData:
    NAME_MAP = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}

    def __init__(self, name, root='./data'):
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_undirected, remove_self_loops

        pyg_name = self.NAME_MAP[name]
        dataset = Planetoid(root=root, name=pyg_name)
        data = dataset[0]

        edge_index = remove_self_loops(data.edge_index)[0]
        self.full_edge_index = to_undirected(edge_index)
        self.node_features = data.x
        self.y = data.y
        self.num_nodes = data.num_nodes
        self.num_features = data.num_features
        self.num_classes = dataset.num_classes

        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask

    def get_train_edge_index(self):
        return self.full_edge_index


class NonCitationData:
    def __init__(self, name, root='./data', train_ratio=0.8, val_ratio=0.1, seed=42):
        from data_loaders import load_dataset
        ds = load_dataset(name, task='node_classification', root=root,
                          train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        self.node_features = ds.node_features
        self.y = ds.y
        self.num_nodes = ds.num_nodes
        self.num_features = ds.num_features
        self.num_classes = ds.num_classes
        self._train_mask = ds.train_mask
        self._val_mask = ds.val_mask
        self._test_mask = ds.test_mask
        self._full_edge_index = ds.full_edge_index

    def get_train_edge_index(self):
        return self._full_edge_index

    @property
    def train_mask(self):
        return self._train_mask

    @property
    def val_mask(self):
        return self._val_mask

    @property
    def test_mask(self):
        return self._test_mask


def load_data(dataset_name, args):
    if dataset_name in CITATION_DATASETS:
        return CitationData(dataset_name, root=args.data_root)
    else:
        return NonCitationData(dataset_name, root=args.data_root, seed=args.seed)


def train(dataset_name, dataset, args, device):
    set_seed(args.seed)

    model = VAEFullModel(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        order=args.order,
        rank=args.rank,
        dropout=args.dropout,
        filter_mode=args.filter_mode,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    x = dataset.node_features.to(device)
    edge_index = dataset.get_train_edge_index().to(device)
    labels = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    is_citation = dataset_name in CITATION_DATASETS
    if is_citation:
        num_classes = dataset.num_classes
        per_class = int(train_mask.sum().item()) // num_classes
        print(f"  Semi-supervised: {train_mask.sum().item()} labeled ({per_class}/class)")
    else:
        print(f"  Split: {train_mask.sum()} train / {val_mask.sum()} val / {test_mask.sum()} test")

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        logits, embeddings, vae_loss = model(x, edge_index)

        loss = criterion(logits[train_mask], labels[train_mask])
        if vae_loss['kl_loss'] is not None:
            loss = loss + args.kl_weight * vae_loss['kl_loss']
        if vae_loss['entropy_loss'] is not None:
            loss = loss + args.entropy_weight * vae_loss['entropy_loss']

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(x, edge_index)
                val_acc = (val_logits[val_mask].argmax(1) == labels[val_mask]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, _, _ = model(x, edge_index)
        test_preds = test_logits[test_mask].argmax(1)
        test_labels = labels[test_mask]

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    preds_np = test_preds.cpu().numpy()
    labels_np = test_labels.cpu().numpy()

    metrics = {
        'accuracy': round(accuracy_score(labels_np, preds_np), 4),
        'f1_macro': round(f1_score(labels_np, preds_np, average='macro', zero_division=0), 4),
        'precision_macro': round(precision_score(labels_np, preds_np, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(labels_np, preds_np, average='macro', zero_division=0), 4),
    }

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f} | Time: {train_time:.1f}s")
    return {'metrics': metrics, 'train_time': round(train_time, 1)}


def main():
    parser = argparse.ArgumentParser(description='VASPE Node Classification')
    parser.add_argument('--datasets', nargs='+', default=None, choices=ALL_DATASETS)
    parser.add_argument('--data-root', default=os.path.join(ROOT_DIR, 'data'))
    parser.add_argument('--output', default=os.path.join(ROOT_DIR, 'results', 'node_classification.json'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--output-dim', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--num-blocks', type=int, default=2)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--filter-mode', default='personalized', choices=['personalized', 'shared'])

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--kl-weight', type=float, default=1e-3)
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Hyperparameters: hidden={args.hidden_dim}, latent={args.latent_dim}, "
          f"blocks={args.num_blocks}, order={args.order}, rank={args.rank}, "
          f"dropout={args.dropout}, lr={args.lr}")

    dataset_names = args.datasets or ALL_DATASETS
    all_results = {}
    total_start = time.time()

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        dataset = load_data(ds_name, args)
        print(f"  Nodes: {dataset.num_nodes} | Features: {dataset.num_features} | Classes: {dataset.num_classes}")

        result = train(ds_name, dataset, args, device)
        all_results[ds_name] = result

    total_time = time.time() - total_start

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'task': 'node_classification', 'model': 'VAEFullModel',
            'device': str(device), 'seed': args.seed,
            'args': vars(args),
            'total_time_seconds': round(total_time, 1),
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  SUMMARY: Node Classification")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Accuracy':>10} {'F1-macro':>10} {'P-macro':>10} {'R-macro':>10} {'Time':>8}")
    print('-' * 70)
    for ds in dataset_names:
        r = all_results[ds]
        print(f"{ds:<20} {r['metrics']['accuracy']:>9.4f} {r['metrics']['f1_macro']:>10.4f} "
              f"{r['metrics']['precision_macro']:>10.4f} {r['metrics']['recall_macro']:>10.4f} {r['train_time']:>6.1f}s")
    print('-' * 70)
    print(f"Total: {total_time:.1f}s | Saved: {args.output}")


if __name__ == '__main__':
    main()
