"""Layer-wise depth features (13)."""
from __future__ import annotations

import numpy as np
import torch


LAYER_NAMES = [
    "vel_5_15", "vel_15_29", "vel_5_29",
    "layer_angle",
    "cos_5_15", "cos_15_29", "cos_5_29",
    "norm_5", "norm_15", "norm_29",
    "norm_ratio_5_29", "norm_ratio_15_29",
    "early_agrees",
]


def build_layer_features(
    h_early: np.ndarray,     # [T, H]  layer ~5
    h_mid: np.ndarray,       # [T, H]  layer ~15
    h_last: np.ndarray,      # [T, H]  final result_norm
    lm_head: np.ndarray,     # [V, H]
    ts: np.ndarray,
    T: int,
) -> np.ndarray:
    h5, h15, h29 = h_early, h_mid, h_last

    vel_5_15 = np.linalg.norm(h15 - h5, axis=-1)
    vel_15_29 = np.linalg.norm(h29 - h15, axis=-1)
    vel_5_29 = np.linalg.norm(h29 - h5, axis=-1)

    dir_5_15 = h15 - h5
    dir_15_29 = h29 - h15
    d1_n = dir_5_15 / (np.linalg.norm(dir_5_15, axis=-1, keepdims=True) + 1e-9)
    d2_n = dir_15_29 / (np.linalg.norm(dir_15_29, axis=-1, keepdims=True) + 1e-9)
    layer_angle = (d1_n * d2_n).sum(axis=-1)

    def cos_sim(a, b):
        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        return (an * bn).sum(axis=-1)

    cos_5_15 = cos_sim(h5, h15)
    cos_15_29 = cos_sim(h15, h29)
    cos_5_29 = cos_sim(h5, h29)

    n5 = np.linalg.norm(h5, axis=-1)
    n15 = np.linalg.norm(h15, axis=-1)
    n29 = np.linalg.norm(h29, axis=-1)
    norm_ratio_5_29 = n5 / (n29 + 1e-6)
    norm_ratio_15_29 = n15 / (n29 + 1e-6)

    with torch.no_grad():
        W = torch.from_numpy(lm_head).to(torch.bfloat16)
        l5 = torch.from_numpy(h5).to(torch.bfloat16) @ W.T
        l29 = torch.from_numpy(h29).to(torch.bfloat16) @ W.T
        argmax_5 = l5.float().argmax(dim=-1).numpy()
        argmax_29 = l29.float().argmax(dim=-1).numpy()
        early_agrees = (argmax_5 == argmax_29).astype(np.float32)

    feat = np.stack([
        np.log1p(vel_5_15[ts + 1]),
        np.log1p(vel_15_29[ts + 1]),
        np.log1p(vel_5_29[ts + 1]),
        layer_angle[ts + 1],
        cos_5_15[ts + 1],
        cos_15_29[ts + 1],
        cos_5_29[ts + 1],
        np.log1p(n5[ts + 1]),
        np.log1p(n15[ts + 1]),
        np.log1p(n29[ts + 1]),
        norm_ratio_5_29[ts + 1],
        norm_ratio_15_29[ts + 1],
        early_agrees[ts + 1],
    ], axis=1).astype(np.float32)
    return feat
