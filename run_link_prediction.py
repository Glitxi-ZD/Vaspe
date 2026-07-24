import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from mymodel import VAEFullModel
from data_loaders import load_dataset

ALL_DATASETS = ['cora', 'citeseer', 'pubmed', 'coauthor_cs', 'wikics', 'amazon_computers']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def precompute_eval_negatives(pos_edges, num_nodes, num_neg=100, seed=123):
    g = torch.Generator()
    g.manual_seed(seed)
    num_pairs = len(pos_edges)
    candidates = torch.randint(0, num_nodes, (num_pairs, num_neg * 5), generator=g, dtype=torch.long)
    neg_samples = []
    for i, (src, dst) in enumerate(pos_edges):
        valid = (candidates[i] != src) & (candidates[i] != dst)
        idx = valid.nonzero(as_tuple=True)[0]
        neg_samples.append(candidates[i, idx[:num_neg]].tolist() if len(idx) >= num_neg
                           else candidates[i, :num_neg].tolist())
    return neg_samples


def evaluate(model, dataset, device, num_neg=100):
    from sklearn.metrics import roc_auc_score, average_precision_score

    model.eval()
    x = dataset.node_features.to(device)
    ei = dataset.get_train_edge_index().to(device)

    with torch.no_grad():
        embeddings, _ = model(x, ei)

    test_edges = dataset.test_edges
    neg_samples = precompute_eval_negatives(test_edges, dataset.num_nodes, num_neg, seed=42)

    pos_scores, all_neg_scores = [], []
    for i, (src, dst) in enumerate(test_edges):
        pos_scores.append(torch.dot(embeddings[src], embeddings[dst]).item())
        neg_emb = embeddings[neg_samples[i]]
        all_neg_scores.append((embeddings[src].unsqueeze(0) @ neg_emb.T).squeeze(0).cpu().numpy())

    pos_scores = np.array(pos_scores)
    all_neg_scores = np.array(all_neg_scores)

    auc_list, ap_list, mrr_list = [], [], []
    hits_10, hits_20, hits_50 = [], [], []

    for i in range(len(test_edges)):
        scores = np.concatenate([[pos_scores[i]], all_neg_scores[i]])
        labels = np.concatenate([[1], np.zeros(num_neg)])

        if len(np.unique(labels)) > 1:
            auc_list.append(roc_auc_score(labels, scores))
            ap_list.append(average_precision_score(labels, scores))
        else:
            auc_list.append(0.5)
            ap_list.append(0.5)

        rank = np.sum(scores >= pos_scores[i])
        mrr_list.append(1.0 / rank)

        sorted_idx = np.argsort(-scores)
        rank_pos = np.where(sorted_idx == 0)[0][0] + 1
        hits_10.append(1.0 if rank_pos <= 10 else 0.0)
        hits_20.append(1.0 if rank_pos <= 20 else 0.0)
        hits_50.append(1.0 if rank_pos <= 50 else 0.0)

    return {
        'AUC': round(float(np.mean(auc_list)), 4),
        'AP': round(float(np.mean(ap_list)), 4),
        'MRR': round(float(np.mean(mrr_list)), 4),
        'Hits@10': round(float(np.mean(hits_10)), 4),
        'Hits@20': round(float(np.mean(hits_20)), 4),
        'Hits@50': round(float(np.mean(hits_50)), 4),
    }


def train(dataset_name, dataset, args, device):
    set_seed(args.seed)

    dataset = load_dataset(dataset_name, task='link_prediction', root=args.data_root,
                           train_ratio=0.8, val_ratio=0.1, seed=args.seed)

    model = VAEFullModel(
        input_dim=dataset.num_features,
        num_classes=None,
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

    x = dataset.node_features.to(device)
    edge_index = dataset.get_train_edge_index().to(device)
    num_nodes = dataset.num_nodes
    train_pairs = torch.tensor(dataset.train_edges, dtype=torch.long, device=device)
    val_pairs = torch.tensor(dataset.val_edges, dtype=torch.long, device=device)

    print(f"  Split: {len(dataset.train_edges)} train / {len(dataset.val_edges)} val / {len(dataset.test_edges)} test")

    best_val_auc = 0.0
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()

        embeddings, vae_loss = model(x, edge_index)

        batch_size = min(256, train_pairs.size(0))
        idx = torch.randperm(train_pairs.size(0), device=device)[:batch_size]
        pos_batch = train_pairs[idx]

        src_emb = embeddings[pos_batch[:, 0]]
        dst_emb = embeddings[pos_batch[:, 1]]
        pos_scores = torch.sum(src_emb * dst_emb, dim=-1)

        neg_dst = torch.randint(0, num_nodes, (batch_size,), device=device)
        neg_emb = embeddings[neg_dst]
        neg_scores = torch.sum(src_emb * neg_emb, dim=-1)

        loss = bpr_loss(pos_scores, neg_scores)

        if vae_loss['kl_loss'] is not None:
            loss = loss + args.kl_weight * vae_loss['kl_loss']
        if vae_loss['entropy_loss'] is not None:
            loss = loss + args.entropy_weight * vae_loss['entropy_loss']

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_emb, _ = model(x, edge_index)
                vp = torch.sum(val_emb[val_pairs[:, 0]] * val_emb[val_pairs[:, 1]], dim=-1)
                vn_dst = torch.randint(0, num_nodes, (val_pairs.size(0),), device=device)
                vn = torch.sum(val_emb[val_pairs[:, 0]] * val_emb[vn_dst], dim=-1)
                val_auc = (vp > vn).float().mean().item()

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
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

    eval_start = time.time()
    metrics = evaluate(model, dataset, device, num_neg=100)
    eval_time = time.time() - eval_start

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"  AUC: {metrics['AUC']:.4f} | AP: {metrics['AP']:.4f} | MRR: {metrics['MRR']:.4f} | Time: {train_time:.1f}s")
    return {'metrics': metrics, 'train_time': round(train_time, 1), 'eval_time': round(eval_time, 1)}


def main():
    parser = argparse.ArgumentParser(description='SSP-GRL Link Prediction')
    parser.add_argument('--datasets', nargs='+', default=None, choices=ALL_DATASETS)
    parser.add_argument('--data-root', default=os.path.join(ROOT_DIR, 'data'))
    parser.add_argument('--output', default=os.path.join(ROOT_DIR, 'results', 'link_prediction.json'))
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

        result = train(ds_name, None, args, device)
        all_results[ds_name] = result

    total_time = time.time() - total_start

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'task': 'link_prediction', 'model': 'VAEFullModel',
            'device': str(device), 'seed': args.seed,
            'args': vars(args),
            'total_time_seconds': round(total_time, 1),
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"  SUMMARY: Link Prediction")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'AUC':>8} {'AP':>8} {'MRR':>8} {'H@10':>8} {'H@20':>8} {'H@50':>8} {'Time':>8}")
    print('-' * 80)
    for ds in dataset_names:
        r = all_results[ds]['metrics']
        t = all_results[ds]['train_time']
        print(f"{ds:<20} {r['AUC']:>7.4f} {r['AP']:>7.4f} {r['MRR']:>7.4f} "
              f"{r['Hits@10']:>7.4f} {r['Hits@20']:>7.4f} {r['Hits@50']:>7.4f} {t:>6.1f}s")
    print('-' * 80)
    print(f"Total: {total_time:.1f}s | Saved: {args.output}")


if __name__ == '__main__':
    main()
