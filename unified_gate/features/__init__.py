"""Feature extraction for the unified-gate skip-scoring MLP.

Ports the 70-feature bundle from MedusaBitNet's test_ultimate_combined.py to
operate on per-sequence arrays instead of file-based memmaps.

Layout (70 features, order matches gate_k20.pt.feature_indices):
    [ 0: 17)  base        17  (content_conf ... conf_min)
    [17: 22)  tier_b       5  (trigram_rep, bigram_freq, vocab_div, dist_same_log, conf_var20)
    [22: 24)  velocity     2  (state_velocity, state_accel)
    [24: 26)  cluster      2  (cluster_mindist, cluster_entropy)
    [26: 28)  moments      2  (softmax_skew, softmax_kurt)
    [28: 30)  hnorm        2  (hidden_norm, norm_drift)
    [30: 32)  fe           2  (free_energy, fe_adjusted)
    [32: 35)  neighborhood 3  (nbr_min/mean/median)
    [35: 43)  holographic  8  (surface/bulk/ratio, corr_len, fisher, uncert, path, ev_horiz)
    [43: 48)  token_reuse  5  (freq, rank, cumcount, heavy_hitter, distinct)
    [48: 52)  phase        4  (phase_c1, c2, c3, residual)
    [52: 55)  rg           3  (rg_div1, div3, div9)
    [55: 57)  superpos     2  (sup_spread, sup_eff_rank)
    [57: 70)  layer_wise  13  (velocities, angles, cosines, norms, ratios, early_agrees)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import (
    BASE_NAMES, build_base_features,
)
from .tier_b import TIER_B_NAMES, build_tier_b
from .physics import (
    PHYSICS_NAMES,
    velocity_accel, cluster, softmax_higher_moments,
    hidden_norm_features, free_energy_analog,
)
from .neighborhood import NEIGHBORHOOD_NAMES, build_neighborhood_features
from .holographic import HOLOGRAPHIC_NAMES, build_holographic_features
from .reuse import REUSE_NAMES, build_token_reuse_features
from .round8 import (
    ROUND8_NAMES,
    aperture_phase_svd, aperture_rg_multiscale, aperture_superposition_structure,
)
from .layer_wise import LAYER_NAMES, build_layer_features


FEATURE_NAMES: list[str] = (
    BASE_NAMES                                          # 17
    + TIER_B_NAMES                                      # 5
    + ["state_velocity", "state_accel"]                 # 2
    + ["cluster_mindist", "cluster_entropy"]            # 2
    + ["softmax_skew", "softmax_kurt"]                  # 2
    + ["hidden_norm", "norm_drift"]                     # 2
    + ["free_energy", "fe_adjusted"]                    # 2
    + NEIGHBORHOOD_NAMES                                # 3
    + HOLOGRAPHIC_NAMES                                 # 8
    + REUSE_NAMES                                       # 5
    + ["phase_c1", "phase_c2", "phase_c3", "phase_residual"]  # 4
    + ["rg_div1", "rg_div3", "rg_div9"]                 # 3
    + ["sup_spread", "sup_eff_rank"]                    # 2
    + LAYER_NAMES                                       # 13
)
N_FEATURES = len(FEATURE_NAMES)  # 70


def _boundary_ids(tokenizer=None, period_id=None, newline_id=None):
    """Return sets of token ids representing sentence-enders and newlines.

    Preferred: pass a HF tokenizer via ``tokenizer`` so we scan full vocab
    (matches training pipeline exactly).
    Fallback: use single IDs ``period_id`` / ``newline_id``.
    """
    if tokenizer is not None:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        enders, newlines = set(), set()
        for tid in range(vocab_size):
            s = tokenizer.decode([tid])
            if "\n" in s:
                newlines.add(tid)
            stripped = s.strip()
            if stripped and stripped[-1] in ".!?":
                enders.add(tid)
        return enders, newlines
    return ({int(period_id)} if period_id is not None else set()), \
           ({int(newline_id)} if newline_id is not None else set())


def extract_all_features(
    hidden_last: np.ndarray,      # [T, H]  final-layer result_norm, float32
    hidden_mid: np.ndarray,       # [T, H]  middle layer (~15), float32
    hidden_early: np.ndarray,     # [T, H]  early layer (~5), float32
    head_logits: np.ndarray,      # [T, K_heads, V]  Medusa head logits, float32
    lm_head: np.ndarray,          # [V, H]  output embedding, float32
    tokens: np.ndarray,           # [T]    token ids
    tokenizer_period_id: int = 13,
    tokenizer_newline_id: int = 198,
    *,
    tokenizer=None,
    period_ids: set | None = None,
    newline_ids: set | None = None,
    cluster_centers: np.ndarray | None = None,
) -> np.ndarray:
    """Extract all 70 features for one sequence.

    Returns a ``[T - 8, 70]`` float32 array. Positions correspond to
    ``ts = arange(6, T - 2)`` (matching the MedusaBitNet training layout).

    Notes:
      - The cluster feature uses K-means centers. If ``cluster_centers`` is
        None, K-means is fit on the hidden states of this sequence — fine for
        typical scoring but for exact reproduction of training-time stats
        users should pass precomputed centers.
      - ``tokenizer_period_id`` / ``tokenizer_newline_id`` are used as fallback
        boundary-token sets when no HF ``tokenizer`` is supplied.
    """
    hidden_last = np.ascontiguousarray(hidden_last, dtype=np.float32)
    hidden_mid = np.ascontiguousarray(hidden_mid, dtype=np.float32)
    hidden_early = np.ascontiguousarray(hidden_early, dtype=np.float32)
    head_logits = np.ascontiguousarray(head_logits, dtype=np.float32)
    lm_head = np.ascontiguousarray(lm_head, dtype=np.float32)
    tokens = np.asarray(tokens)

    T, H = hidden_last.shape
    if T < 10:
        raise ValueError(f"sequence too short: T={T}, need >=10")

    valid = T - 2
    ts = np.arange(6, valid, dtype=np.int64)

    if period_ids is None or newline_ids is None:
        p_ids, n_ids = _boundary_ids(
            tokenizer=tokenizer,
            period_id=tokenizer_period_id, newline_id=tokenizer_newline_id,
        )
        period_ids = p_ids if period_ids is None else period_ids
        newline_ids = n_ids if newline_ids is None else newline_ids

    # --- base 17 ---
    base_feat, _label, confs0, probs0 = build_base_features(
        head_logits, hidden_last, lm_head, tokens,
        ts, T, valid, period_ids, newline_ids,
    )

    # --- tier_b 5 ---
    tb = build_tier_b(tokens, confs0, ts, T)

    # --- velocity/accel 2 ---
    va = velocity_accel(hidden_last, ts, T)

    # --- cluster 2 ---
    cl, _ = cluster(hidden_last, ts, T, centers=cluster_centers)

    # --- moments 2 ---
    V = head_logits.shape[-1]
    mo = softmax_higher_moments(probs0, ts, V)

    # --- hnorm 2 ---
    hn = hidden_norm_features(hidden_last, ts, T)

    # --- free energy 2 ---
    fe = free_energy_analog(cl)

    # --- neighborhood 3 ---
    nbr = build_neighborhood_features(hidden_last, ts, T)

    # --- holographic 8 ---
    hol = build_holographic_features(head_logits, hidden_last, probs0, ts, T)

    # --- token reuse 5 ---
    tr = build_token_reuse_features(tokens, ts, T)

    # --- phase 4 ---
    phase = aperture_phase_svd(hidden_last, ts, T)

    # --- rg 3 ---
    rg = aperture_rg_multiscale(hidden_last, ts, T)

    # --- superposition 2 ---
    sup = aperture_superposition_structure(probs0, lm_head, ts, T)

    # --- layer-wise 13 ---
    layer = build_layer_features(hidden_early, hidden_mid, hidden_last, lm_head, ts, T)

    X = np.concatenate(
        [base_feat, tb, va, cl, mo, hn, fe, nbr, hol, tr, phase, rg, sup, layer],
        axis=1,
    ).astype(np.float32)
    assert X.shape[1] == N_FEATURES, (X.shape, N_FEATURES)
    return X


__all__ = ["extract_all_features", "FEATURE_NAMES", "N_FEATURES"]
