"""Hidden-state neighborhood features (3): min/mean/median distance to recent states."""
from __future__ import annotations

import numpy as np
import torch


NEIGHBORHOOD_NAMES = ["nbr_min_dist", "nbr_mean_dist", "nbr_median_dist"]


def build_neighborhood_features(
    hidden_last: np.ndarray, ts: np.ndarray, T: int, fifo_size: int = 50
) -> np.ndarray:
    h_t = torch.from_numpy(hidden_last)
    feat_seq = np.zeros((T, 3), dtype=np.float32)
    for i in range(1, T):
        lo = max(0, i - fifo_size)
        fifo = h_t[lo:i]
        cur = h_t[i:i + 1]
        dists = torch.cdist(cur, fifo).squeeze(0)
        if len(dists) == 0:
            feat_seq[i] = [0, 0, 0]
        else:
            feat_seq[i, 0] = float(dists.min())
            feat_seq[i, 1] = float(dists.mean())
            feat_seq[i, 2] = float(torch.median(dists))
    return feat_seq[ts + 1].astype(np.float32)
