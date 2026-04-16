"""Holographic / info-geometry features (8): surface/bulk entropy + ratio, correlation length,
fisher proxy, uncertainty product, path integral, event horizon."""
from __future__ import annotations

import numpy as np
import torch


HOLOGRAPHIC_NAMES = [
    "surface_ent", "bulk_ent", "sb_ratio",
    "corr_len",
    "fisher_proxy",
    "uncert_product",
    "path_integral",
    "event_horizon",
]


def build_holographic_features(
    head_logits: np.ndarray,     # [T, K, V]
    hidden_last: np.ndarray,     # [T, H]
    probs0_full: torch.Tensor,   # [T, V] head-0 softmax
    ts: np.ndarray,
    T: int,
    surface_k: int = 10,
    bulk_k: int = 50,
) -> np.ndarray:
    import torch.nn.functional as F

    probs = probs0_full
    entropy = -(probs * probs.log().clamp_min(-20)).sum(-1).numpy()
    conf = probs.max(-1).values.numpy()
    log_prob_of_argmax = probs.max(-1).values.log().clamp_min(-20).numpy()
    logits0 = torch.from_numpy(head_logits[:, 0, :]).float()

    surface_ent = np.zeros(T, dtype=np.float32)
    bulk_ent = np.zeros(T, dtype=np.float32)
    sb_ratio = np.zeros(T, dtype=np.float32)
    for i in range(bulk_k, T):
        surf = entropy[i - surface_k:i]
        bulk = entropy[i - bulk_k:i - surface_k]
        surface_ent[i] = surf.mean()
        bulk_ent[i] = bulk.mean() if len(bulk) > 0 else 0
        sb_ratio[i] = surface_ent[i] / (bulk_ent[i] + 1e-6)

    h_np = hidden_last
    correlation_len = np.zeros(T, dtype=np.float32)
    for i in range(bulk_k, T):
        early = h_np[i - bulk_k:i - bulk_k // 2]
        late = h_np[i - bulk_k // 2:i]
        en = np.linalg.norm(early, axis=-1)
        ln_ = np.linalg.norm(late, axis=-1)
        if len(en) > 1 and len(ln_) > 1:
            minlen = min(len(en), len(ln_))
            c = np.corrcoef(en[:minlen], ln_[:minlen])[0, 1]
            correlation_len[i] = c if np.isfinite(c) else 0

    top5_logits = torch.topk(logits0, 5, dim=-1).values
    fisher_proxy = top5_logits.std(dim=-1).numpy()

    conf_var_local = np.zeros(T, dtype=np.float32)
    for i in range(10, T):
        conf_var_local[i] = conf[i - 10:i].std()
    uncert_product = conf * conf_var_local

    path_integral_ = np.zeros(T, dtype=np.float32)
    for i in range(20, T):
        path_integral_[i] = log_prob_of_argmax[i - 20:i].sum()

    ev_horiz = np.full(T, 200.0, dtype=np.float32)
    last_confident = -1
    for i in range(T):
        if last_confident >= 0:
            d = np.linalg.norm(h_np[i] - h_np[last_confident])
            ev_horiz[i] = min(200, d)
        if conf[i] > 0.8:
            last_confident = i

    feat = np.stack([
        surface_ent[ts + 1],
        bulk_ent[ts + 1],
        sb_ratio[ts + 1],
        correlation_len[ts + 1],
        fisher_proxy[ts + 1],
        uncert_product[ts + 1],
        path_integral_[ts + 1],
        np.log1p(ev_horiz[ts + 1]),
    ], axis=1).astype(np.float32)
    return feat
