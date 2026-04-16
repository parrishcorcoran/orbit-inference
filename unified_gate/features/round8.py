"""Round-8 physics apertures: phase-SVD (4), RG (3), superposition (2) = 9 total."""
from __future__ import annotations

import numpy as np
import torch


ROUND8_NAMES = [
    # phase (4)
    "phase_c1", "phase_c2", "phase_c3", "phase_residual",
    # rg (3)
    "rg_div1", "rg_div3", "rg_div9",
    # superposition (2)
    "sup_spread", "sup_eff_rank",
]


def aperture_phase_svd(hidden_last: np.ndarray, ts: np.ndarray, T: int,
                       window: int = 20, n_components: int = 3) -> np.ndarray:
    h = hidden_last
    phase_coords = np.zeros((T, n_components), dtype=np.float32)
    residual = np.zeros(T, dtype=np.float32)
    for i in range(window, T):
        win = h[i - window:i]
        win_c = win - win.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(win_c, full_matrices=False)
        h_cur = h[i] - win.mean(axis=0)
        coords = h_cur @ Vt[:n_components].T
        phase_coords[i] = coords
        reconstruct = coords @ Vt[:n_components]
        residual[i] = np.linalg.norm(h_cur - reconstruct)

    feat = np.concatenate([
        phase_coords[ts + 1],
        residual[ts + 1].reshape(-1, 1),
    ], axis=1).astype(np.float32)
    return feat


def aperture_rg_multiscale(hidden_last: np.ndarray, ts: np.ndarray, T: int) -> np.ndarray:
    h = hidden_last
    div_1 = np.zeros(T, dtype=np.float32)
    div_3 = np.zeros(T, dtype=np.float32)
    div_9 = np.zeros(T, dtype=np.float32)
    for i in range(10, T):
        h_cur = h[i]
        div_1[i] = np.linalg.norm(h_cur - h[i - 1])
        div_3[i] = np.linalg.norm(h_cur - h[max(0, i - 3):i].mean(axis=0))
        div_9[i] = np.linalg.norm(h_cur - h[max(0, i - 9):i].mean(axis=0))
    return np.stack([div_1[ts + 1], div_3[ts + 1], div_9[ts + 1]], axis=1).astype(np.float32)


def aperture_superposition_structure(
    probs0_full: torch.Tensor,   # [T, V]
    lm_head: np.ndarray,         # [V, H]
    ts: np.ndarray,
    T: int,
    K: int = 32,
) -> np.ndarray:
    probs = probs0_full
    top = torch.topk(probs, K, dim=-1)
    top_k_idx = top.indices.numpy()
    top_k_probs = top.values.numpy()

    W = torch.from_numpy(lm_head).to(torch.bfloat16)
    tok_spread = np.zeros(T, dtype=np.float32)
    eff_rank = np.zeros(T, dtype=np.float32)
    for i in range(T):
        idx = top_k_idx[i]
        tok_embs = W[idx].float().numpy()
        centroid = tok_embs.mean(axis=0)
        spread = np.linalg.norm(tok_embs - centroid, axis=-1).mean()
        tok_spread[i] = spread
        p = top_k_probs[i] / top_k_probs[i].sum()
        h_ent = -np.sum(p * np.log(p + 1e-12))
        eff_rank[i] = np.exp(h_ent)
    # Note: parent uses ts (NOT ts+1) for superposition
    return np.stack([tok_spread[ts], eff_rank[ts]], axis=1).astype(np.float32)
