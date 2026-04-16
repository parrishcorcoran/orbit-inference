"""Minimal-feature-set ablation.

After the 70-feature saturation + cross-model finding that per-seq intrinsic
dim is ~7, the bundle is almost certainly over-parameterized. Goal: find the
knee of the skip/K curve. The smallest K that hits ~95% of the 70-feature
frontier is the deployment target.

Procedure:
  1. Build the full 70-feature matrix (same as test_ultimate_combined.py).
  2. Train the 70-feature MLP, score per-feature gradient importance on held-out.
  3. Take top-K by importance for K in {5, 7, 10, 15, 20, 30, 50}.
  4. Retrain each K-feature MLP and measure the frontier.
  5. Print table.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import (
    aperture_velocity_and_accel, aperture_cluster,
)
from test_physics_more import (
    softmax_higher_moments, hidden_state_norm_features, free_energy_analog,
)
from test_hidden_neighborhood import build_neighborhood_features
from test_holographic import build_holographic_features
from test_tier_b_features import build_tier_b
from test_token_reuse import build_token_reuse_features
from test_physics_round8 import (
    aperture_phase_svd, aperture_superposition_structure, aperture_rg_multiscale,
)
from test_layer_wise import build_layer_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads


SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def build_all_features():
    """Same as test_ultimate_combined — 70 features."""
    import torch.nn.functional as F
    X_base, y, train_mask, test_mask = build_features()
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    confs_full = []
    with torch.no_grad():
        for si in range(48):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h_raw = chunk.astype(np.uint32) << 16
            h = h_raw.view(np.float32).reshape(SEQ_LEN, HIDDEN)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            cf = F.softmax(logits, dim=-1).max(-1).values.numpy()
            confs_full.append(cf)

    print("Building feature groups...")
    tier_b = build_tier_b(tokens_mm, confs_full)
    va = aperture_velocity_and_accel()
    cl = aperture_cluster()
    mo = softmax_higher_moments()
    hn = hidden_state_norm_features()
    fe = free_energy_analog(cl)
    nbr = build_neighborhood_features()
    hol = build_holographic_features()
    tr = build_token_reuse_features()
    phase = aperture_phase_svd()
    rg = aperture_rg_multiscale()
    sup = aperture_superposition_structure()
    layer = build_layer_features()

    # Track group boundaries and name each feature roughly
    groups = [
        ("base", X_base, ["content_conf","content_entropy","logit_gap","purity",
            "top3_cov","top10_cov","rc10","rc50","conf_deriv","conf_lag1",
            "conf_lag5","dist_period_log","dist_newline_log","rel_pos",
            "agreement_count","conf_var","conf_min"]),
        ("tier_b", tier_b, [f"tierb_{i}" for i in range(tier_b.shape[1])]),
        ("velocity", va, [f"vel_{i}" for i in range(va.shape[1])]),
        ("cluster", cl, [f"cluster_{i}" for i in range(cl.shape[1])]),
        ("moments", mo, [f"mom_{i}" for i in range(mo.shape[1])]),
        ("hnorm", hn, [f"hnorm_{i}" for i in range(hn.shape[1])]),
        ("fe", fe, [f"fe_{i}" for i in range(fe.shape[1])]),
        ("nbr", nbr, [f"nbr_{i}" for i in range(nbr.shape[1])]),
        ("hol", hol, [f"hol_{i}" for i in range(hol.shape[1])]),
        ("treuse", tr, [f"treuse_{i}" for i in range(tr.shape[1])]),
        ("phase", phase, [f"phase_{i}" for i in range(phase.shape[1])]),
        ("rg", rg, [f"rg_{i}" for i in range(rg.shape[1])]),
        ("sup", sup, [f"sup_{i}" for i in range(sup.shape[1])]),
        ("layer", layer, [f"layer_{i}" for i in range(layer.shape[1])]),
    ]
    X_all = np.concatenate([g[1] for g in groups], axis=1)
    names = [n for g in groups for n in g[2]]
    assert len(names) == X_all.shape[1]
    return X_all, y, train_mask, test_mask, names


def train_mlp(X_all, y, train_mask, test_mask, feat_idx, hidden=128, epochs=60, seed=0):
    mu = X_all[train_mask][:, feat_idx].mean(axis=0)
    sd = X_all[train_mask][:, feat_idx].std(axis=0) + 1e-6
    Xn = (X_all[:, feat_idx] - mu) / sd
    torch.manual_seed(seed)
    net = UnifiedMLP(n_feat=len(feat_idx), hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    Xe = torch.from_numpy(Xn[test_mask])
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            bi = perm[i:i+4096]
            loss = F.binary_cross_entropy_with_logits(net(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xe)).numpy()
    return net, Xn, p


def main():
    X_all, y, train_mask, test_mask, names = build_all_features()
    n = X_all.shape[1]
    print(f"\nTotal features: {n}")
    assert len(names) == n

    # Train full model, compute gradient importance
    print("\nTraining 70-feature model for importance ranking...")
    net, Xn, p = train_mlp(X_all, y, train_mask, test_mask, list(range(n)))
    print("Frontier (full 70 features):")
    full_frontier = list(frontier(p, y[test_mask], LAMBDA_TARGETS))
    for lam, skip, fid in full_frontier:
        print(f"  λ={lam:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    order = np.argsort(-grads)

    print("\n=== Top 20 by |grad| ===")
    for i, idx in enumerate(order[:20]):
        print(f"  {i+1:2d}. {names[idx]:>22}  |grad|={grads[idx]:.4f}")

    # Ablate at K in {5, 7, 10, 15, 20, 30, 50}
    results = {}
    Ks = [5, 7, 10, 15, 20, 30, 50, 70]
    print(f"\n{'K':>3}  " + "  ".join(f"λ={lam:.2f}" for lam, _, _ in full_frontier))
    print("-" * 60)
    for K in Ks:
        top_k_idx = order[:K].tolist()
        _, _, p_k = train_mlp(X_all, y, train_mask, test_mask, top_k_idx,
                              hidden=min(128, max(32, K * 2)), epochs=60, seed=0)
        fr = list(frontier(p_k, y[test_mask], LAMBDA_TARGETS))
        results[K] = fr
        row = f"{K:>3d}  " + "  ".join(f"skip={s:.4f}" for _, s, _ in fr)
        print(row)

    print("\n=== Fidelities ===")
    for K in Ks:
        row = f"{K:>3d}  " + "  ".join(f"fid={f:.4f}" for _, _, f in results[K])
        print(row)

    # Print knee suggestion
    print("\n=== Analysis ===")
    base_skip = {lam: s for lam, s, _ in full_frontier}
    for K in Ks:
        for (lam, skip, fid) in results[K]:
            if lam == 0.95:
                frac = skip / base_skip[lam] * 100
                print(f"  K={K:2d}: skip at λ=0.95 = {skip:.4f} ({frac:.1f}% of full)")


if __name__ == "__main__":
    main()
