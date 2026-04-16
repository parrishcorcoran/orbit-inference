"""Cross-model intrinsic dim: Llama 3.1 8B last-layer hidden states.

Llama 8B data is stored as raw float32 (not bfloat16-as-uint16 like BitNet),
and has hidden=4096 (vs 2560 for BitNet).

Research question: is the 14-dim intrinsic manifold a property of the token
stream, or of BitNet specifically? If Llama 8B also lands near 14, the
framework generalizes — which is the thesis claim.
"""
import numpy as np
import torch
import time

SEQ_LEN = 2048
HIDDEN = 4096
N_SAMPLES = 5000
N_BOOTSTRAPS = 5


def twonn_mle(points):
    N, D = points.shape
    dists = torch.cdist(points, points)
    dists.fill_diagonal_(float('inf'))
    vals, _ = torch.topk(dists, k=2, dim=1, largest=False)
    r1 = vals[:, 0]; r2 = vals[:, 1]
    valid = (r1 > 1e-6) & (r2 > r1)
    r1 = r1[valid]; r2 = r2[valid]
    mu = r2 / r1
    return len(mu) / torch.log(mu).sum().item(), len(mu)


def main():
    mm = np.memmap("data/hidden_llama8b_layer31.bin", dtype=np.float32, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_seqs = mm.size // per_seq
    print(f"Llama 8B: {n_seqs} seqs × {SEQ_LEN} tokens × {HIDDEN} hidden "
          f"= {n_seqs*SEQ_LEN:,} points")

    rng = np.random.default_rng(42)
    estimates = []
    for boot in range(N_BOOTSTRAPS):
        print(f"\n=== Bootstrap {boot+1}/{N_BOOTSTRAPS} ===")
        t0 = time.time()
        seq_idxs = rng.integers(0, n_seqs, N_SAMPLES)
        pos_idxs = rng.integers(0, SEQ_LEN, N_SAMPLES)
        flat = seq_idxs * SEQ_LEN + pos_idxs

        pts = np.zeros((N_SAMPLES, HIDDEN), dtype=np.float32)
        for i, fi in enumerate(flat):
            s = fi * HIDDEN
            pts[i] = mm[s:s + HIDDEN]
        t1 = time.time(); print(f"  sampled in {t1-t0:.1f}s")
        d_hat, n_val = twonn_mle(torch.from_numpy(pts))
        print(f"  TwoNN MLE dim = {d_hat:.2f}  (valid {n_val}/{N_SAMPLES}, "
              f"{time.time()-t1:.1f}s)")
        estimates.append(d_hat)

    e = np.array(estimates)
    print(f"\n*** Llama 3.1 8B layer-31 intrinsic dim ***")
    print(f"  mean   = {e.mean():.2f}")
    print(f"  median = {np.median(e):.2f}")
    print(f"  std    = {e.std():.2f}")
    print(f"  range  = [{e.min():.2f}, {e.max():.2f}]")
    print(f"\nFor comparison: BitNet 2B layer-29 (result_norm) = 14.40")
    print(f"Same-tokenizer cross-model test — if similar, framework is "
          f"model-agnostic.")


if __name__ == "__main__":
    main()
