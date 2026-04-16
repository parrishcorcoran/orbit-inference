"""Physics apertures: velocity/accel (2), cluster (2), moments (2), hnorm (2), FE (2)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


PHYSICS_NAMES = [
    "state_velocity", "state_accel",
    "cluster_mindist", "cluster_entropy",
    "softmax_skew", "softmax_kurt",
    "hidden_norm", "norm_drift",
    "free_energy", "fe_adjusted",
]


def velocity_accel(hidden_last: np.ndarray, ts: np.ndarray, T: int) -> np.ndarray:
    h_t = torch.from_numpy(hidden_last)
    vel = torch.zeros(T)
    vel[1:] = (h_t[1:] - h_t[:-1]).norm(dim=-1)
    accel = torch.zeros(T)
    accel[2:] = (h_t[2:] - 2 * h_t[1:-1] + h_t[:-2]).norm(dim=-1)
    return np.stack([vel[ts + 1].numpy(), accel[ts + 1].numpy()], axis=1).astype(np.float32)


def _kmeans(data: torch.Tensor, K: int, n_iter: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    N = len(data)
    idx0 = rng.choice(N, size=K, replace=False)
    centers = data[idx0].clone()
    for _ in range(n_iter):
        dists = torch.cdist(data, centers)
        assigns = dists.argmin(dim=-1)
        for k in range(K):
            mask = assigns == k
            if mask.sum() > 0:
                centers[k] = data[mask].mean(dim=0)
    return centers


def cluster(
    hidden_last: np.ndarray,
    ts: np.ndarray,
    T: int,
    K: int = 32,
    centers: np.ndarray | None = None,
    max_train: int = 20000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to nearest K-means center + soft-cluster entropy.

    If ``centers`` is None, fits K-means on this sequence's hidden states.
    Returns (features[T_valid, 2], centers[K, H]) so callers can reuse centers.
    """
    h_t = torch.from_numpy(hidden_last).float()
    if centers is None:
        rng = np.random.default_rng(seed)
        N = len(h_t)
        if N > max_train:
            sub = rng.choice(N, size=max_train, replace=False)
            kmeans_data = h_t[sub]
        else:
            kmeans_data = h_t
        centers_t = _kmeans(kmeans_data, K, seed=seed)
    else:
        centers_t = torch.from_numpy(centers).float()

    dists = torch.cdist(h_t, centers_t)  # [T, K]
    min_d, _ = dists.min(dim=-1)
    soft = F.softmax(-dists * 0.01, dim=-1)
    cluster_entropy = -(soft * torch.log(soft.clamp_min(1e-12))).sum(-1)

    feat = np.stack([
        min_d[ts + 1].numpy(),
        cluster_entropy[ts + 1].numpy(),
    ], axis=1).astype(np.float32)
    return feat, centers_t.numpy()


def softmax_higher_moments(probs0_full: torch.Tensor, ts: np.ndarray, vocab: int) -> np.ndarray:
    """3rd and 4th central moments of head-0 softmax over token indices.

    ``probs0_full`` is [T, V] float.
    """
    token_idx = torch.arange(vocab).float()
    probs = probs0_full
    mean_tok = (probs * token_idx).sum(-1)
    dev = token_idx.unsqueeze(0) - mean_tok.unsqueeze(1)
    var = (probs * dev ** 2).sum(-1)
    m3 = (probs * dev ** 3).sum(-1)
    m4 = (probs * dev ** 4).sum(-1)
    std = var.sqrt().clamp_min(1e-6)
    skew = m3 / std ** 3
    kurt = m4 / std ** 4
    return np.stack([skew[ts].numpy(), kurt[ts].numpy()], axis=1).astype(np.float32)


def hidden_norm_features(hidden_last: np.ndarray, ts: np.ndarray, T: int) -> np.ndarray:
    h_t = torch.from_numpy(hidden_last)
    norms = h_t.norm(dim=-1).numpy()
    drift = np.zeros(T, dtype=np.float32)
    drift[1:] = np.abs(norms[1:] - norms[:-1])
    return np.stack([
        np.log1p(norms[ts + 1]),
        drift[ts + 1],
    ], axis=1).astype(np.float32)


def free_energy_analog(cluster_feats: np.ndarray) -> np.ndarray:
    mindist = cluster_feats[:, 0]
    entropy = cluster_feats[:, 1]
    free_energy = np.log1p(mindist * 0.01).astype(np.float32)
    fe_adjusted = (free_energy - entropy * 0.1).astype(np.float32)
    return np.stack([free_energy, fe_adjusted], axis=1)
