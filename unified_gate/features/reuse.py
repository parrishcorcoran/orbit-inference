"""Token-reuse features (5): freq, rank, cumcount, heavy_hitter, distinct."""
from __future__ import annotations

from collections import Counter

import numpy as np


REUSE_NAMES = [
    "token_freq_window", "token_rank_window", "token_cumcount",
    "is_heavy_hitter", "distinct_in_window",
]


def build_token_reuse_features(tokens: np.ndarray, ts: np.ndarray, T: int, window: int = 100) -> np.ndarray:
    toks = tokens
    freq_in_window = np.zeros(T, dtype=np.float32)
    rank_in_window = np.zeros(T, dtype=np.float32)
    cum_count = np.zeros(T, dtype=np.float32)
    is_heavy_hitter = np.zeros(T, dtype=np.float32)
    distinct_in_window = np.zeros(T, dtype=np.float32)

    total_counter = Counter()
    for i in range(T):
        lo = max(0, i - window + 1)
        window_toks = toks[lo:i + 1]
        window_counter = Counter(window_toks.tolist())
        tid = int(toks[i])
        freq_in_window[i] = window_counter[tid]
        distinct_in_window[i] = len(window_counter)

        sorted_counts = sorted(window_counter.values(), reverse=True)
        try:
            rank = sorted_counts.index(window_counter[tid]) + 1
        except ValueError:
            rank = 1
        rank_in_window[i] = rank

        n_unique = len(sorted_counts)
        threshold_rank = max(1, n_unique // 10)
        is_heavy_hitter[i] = 1.0 if rank <= threshold_rank else 0.0

        total_counter[tid] += 1
        cum_count[i] = total_counter[tid]

    feat = np.stack([
        np.log1p(freq_in_window[ts + 1]),
        np.log1p(rank_in_window[ts + 1]),
        np.log1p(cum_count[ts + 1]),
        is_heavy_hitter[ts + 1],
        np.log1p(distinct_in_window[ts + 1]),
    ], axis=1).astype(np.float32)
    return feat
