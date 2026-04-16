"""Seed-robustness of K=30 beats K=70 finding.

The previous ablation used seed=0 for all K. Here we sweep seeds ∈ {0..4}
for K ∈ {15, 20, 25, 30, 40, 50, 70} and report mean ± std of skip at
λ=0.95. The claim "K=30 is +17% over K=70" needs to survive seed variance.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_minimal_feature_set import build_all_features, train_mlp
from test_unified_estimator_v2 import frontier, LAMBDA_TARGETS


def main():
    X_all, y, train_mask, test_mask, names = build_all_features()
    n = X_all.shape[1]

    # Rank features by importance from a single reference model (seed=0)
    print("Computing feature importance from reference model...")
    net, Xn, _ = train_mlp(X_all, y, train_mask, test_mask, list(range(n)),
                           hidden=128, epochs=60, seed=0)
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    order = np.argsort(-grads)

    Ks = [15, 20, 25, 30, 40, 50, 70]
    seeds = [0, 1, 2, 3, 4]
    print(f"\nSweeping K × seed ({len(Ks)} × {len(seeds)} = {len(Ks)*len(seeds)} models)...\n")

    results = {K: [] for K in Ks}
    for K in Ks:
        feat_idx = order[:K].tolist()
        for seed in seeds:
            hidden = min(128, max(32, K * 2))
            _, _, p = train_mlp(X_all, y, train_mask, test_mask, feat_idx,
                                hidden=hidden, epochs=60, seed=seed)
            fr = list(frontier(p, y[test_mask], LAMBDA_TARGETS))
            skip_095 = [s for lam, s, _ in fr if lam == 0.95][0]
            results[K].append(skip_095)
            print(f"  K={K:3d}  seed={seed}  skip@λ=0.95 = {skip_095:.4f}")

    print("\n=== Summary (skip@λ=0.95 over 5 seeds) ===")
    print(f"{'K':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for K in Ks:
        arr = np.array(results[K])
        print(f"{K:>4}  {arr.mean():>8.4f}  {arr.std():>8.4f}  "
              f"{arr.min():>8.4f}  {arr.max():>8.4f}")


if __name__ == "__main__":
    main()
