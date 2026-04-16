"""Tier-B cheap token-derived features (5)."""
from __future__ import annotations

import numpy as np


TIER_B_NAMES = ["trigram_rep", "bigram_freq", "vocab_div", "dist_same_log", "conf_var20"]


def build_tier_b(tokens: np.ndarray, confs0: np.ndarray, ts: np.ndarray, T: int) -> np.ndarray:
    toks = tokens
    # Trigram repetition in last 100
    trigram_rep = np.zeros(T, dtype=np.float32)
    for i in range(3, T):
        tri = (int(toks[i - 3]), int(toks[i - 2]), int(toks[i - 1]))
        window_start = max(0, i - 100)
        seen = False
        for j in range(window_start, i - 2):
            if (int(toks[j]), int(toks[j + 1]), int(toks[j + 2])) == tri:
                seen = True
                break
        trigram_rep[i] = 1.0 if seen else 0.0

    bigram_freq = np.zeros(T, dtype=np.float32)
    for i in range(2, T):
        bi = (int(toks[i - 2]), int(toks[i - 1]))
        window_start = max(0, i - 100)
        count = 0
        for j in range(window_start, i - 1):
            if (int(toks[j]), int(toks[j + 1])) == bi:
                count += 1
        bigram_freq[i] = count / max(1, i - window_start)

    vocab_div = np.zeros(T, dtype=np.float32)
    for i in range(T):
        lo = max(0, i - 50)
        window = toks[lo:i + 1]
        if len(window) > 0:
            uniq = len(set(window.tolist()))
            vocab_div[i] = uniq / len(window)

    last_seen = {}
    dist_same = np.full(T, 200.0, dtype=np.float32)
    for i in range(T):
        tid = int(toks[i])
        if tid in last_seen:
            dist_same[i] = min(200, i - last_seen[tid])
        last_seen[tid] = i
    dist_same_log = np.log1p(dist_same)

    conf_var20 = np.zeros(T, dtype=np.float32)
    for i in range(T):
        lo = max(0, i - 20)
        conf_var20[i] = float(np.var(confs0[lo:i + 1])) if i > lo else 0.0

    feat = np.stack([
        trigram_rep[ts + 1],
        bigram_freq[ts + 1],
        vocab_div[ts + 1],
        dist_same_log[ts + 1],
        conf_var20[ts + 1],
    ], axis=1).astype(np.float32)
    return feat
