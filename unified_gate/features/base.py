"""Baseline 17 features (ported from MedusaBitNet test_feature_redundancy:build_features)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


BASE_NAMES = [
    "content_conf", "content_entropy",
    "logit_gap", "purity", "top3_cov", "top10_cov",
    "rc10", "rc50", "conf_deriv",
    "conf_lag1", "conf_lag5",
    "dist_period_log", "dist_newline_log", "rel_pos",
    "agreement_count", "conf_var", "conf_min",
]


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def lagged_diff(arr: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(arr)
    out[lag:] = arr[lag:] - arr[:-lag]
    return out


def dist_to_last(token_ids, match_set) -> np.ndarray:
    n = len(token_ids)
    out = np.full(n, 9999, dtype=np.int32)
    last = -10000
    for i in range(n):
        if last >= 0:
            out[i] = i - last
        if int(token_ids[i]) in match_set:
            last = i
    return out


def build_base_features(
    head_logits: np.ndarray,     # [T, K_heads, V]
    hidden_last: np.ndarray,     # [T, H]  result_norm
    lm_head: np.ndarray,         # [V, H]
    tokens: np.ndarray,          # [T]
    ts: np.ndarray,
    T: int,
    valid: int,
    period_ids,
    newline_ids,
):
    """Compute 17 baseline features at positions ``ts`` (=[6..valid)).

    Returns (features[T_valid, 17], label[T_valid]).
    Label = head-0 argmax agrees with verifier (lm_head @ h_{t+1}) at ts.
    """
    # Head-0 distribution
    logits0 = torch.from_numpy(head_logits[:, 0, :]).float()
    probs0 = F.softmax(logits0, dim=-1)
    confs0 = probs0.max(dim=-1).values.numpy()
    top2 = torch.topk(logits0, 2, dim=-1).values
    logit_gap = (top2[:, 0] - top2[:, 1]).numpy()
    purity = (probs0 ** 2).sum(dim=-1).numpy()
    top3_cov = torch.topk(probs0, 3, dim=-1).values.sum(dim=-1).numpy()
    top10_cov = torch.topk(probs0, 10, dim=-1).values.sum(dim=-1).numpy()
    h0_entropy = -(probs0 * torch.log(probs0.clamp_min(1e-12))).sum(-1).numpy()

    # All-head info
    logits_all = torch.from_numpy(head_logits).float()  # [T, K, V]
    probs_all = F.softmax(logits_all, dim=-1)
    confs_all = probs_all.max(dim=-1).values.numpy()    # [T, K]
    preds_all = probs_all.argmax(dim=-1).numpy()        # [T, K]

    # Verifier predictions via lm_head on h[1:]
    h_t = torch.from_numpy(hidden_last).to(torch.bfloat16)
    W = torch.from_numpy(lm_head).to(torch.bfloat16)
    vlogits = h_t[1:] @ W.T  # [T-1, V]
    vpred = vlogits.float().argmax(dim=-1).numpy()  # [T-1]

    # Features at ts
    content_conf = confs0[ts]
    content_entropy = h0_entropy[ts]
    rc10 = rolling_mean(confs0, 10)[ts]
    rc50 = rolling_mean(confs0, 50)[ts]
    conf_deriv = rc10 - rc50
    conf_lag1 = lagged_diff(confs0, 1)[ts]
    conf_lag5 = lagged_diff(confs0, 5)[ts]

    de = dist_to_last(tokens.tolist(), period_ids).astype(np.float32)
    dn = dist_to_last(tokens.tolist(), newline_ids).astype(np.float32)
    de_at = np.log1p(np.minimum(de[ts + 1], 200.0))
    dn_at = np.log1p(np.minimum(dn[ts + 1], 200.0))
    rel_pos = (ts.astype(np.float32) / valid)

    # Agreement across heads; head-k uses shift of k (head 0 at ts, head 1 at ts-1, ...)
    K_heads = preds_all.shape[1]
    h0p = preds_all[:, 0][ts]
    head_preds = [h0p]
    head_confs = [confs_all[:, 0][ts]]
    for k in range(1, min(4, K_heads)):
        head_preds.append(preds_all[:, k][ts - k])
        head_confs.append(confs_all[:, k][ts - k])
    # Pad with head-0 if fewer than 4 heads (preserve shape of agreement/var/min over 4)
    while len(head_preds) < 4:
        head_preds.append(head_preds[0])
        head_confs.append(head_confs[0])

    # NOTE: the training pipeline computed this as ``((h1==h0)+(h2==h0)+(h3==h0))
    # .astype(np.float32)`` — under numpy 2.x bool+bool stays bool (logical OR),
    # so this feature is secretly a 0/1 "at least one head agrees" indicator,
    # not a 0-3 count. We reproduce that behavior here to stay faithful to the
    # trained gate's feature distribution. A corrected retraining lives on the
    # v0.3 roadmap (feature name stays "agreement_count" for artifact
    # compatibility).
    agreement = (
        (head_preds[1] == head_preds[0])
        + (head_preds[2] == head_preds[0])
        + (head_preds[3] == head_preds[0])
    ).astype(np.float32)
    conf_stack = np.stack(head_confs[:4], axis=1)
    conf_var = conf_stack.var(axis=1)
    conf_min = conf_stack.min(axis=1)

    feat = np.stack([
        content_conf, content_entropy,
        logit_gap[ts], purity[ts], top3_cov[ts], top10_cov[ts],
        rc10, rc50, conf_deriv,
        conf_lag1, conf_lag5,
        de_at, dn_at, rel_pos,
        agreement, conf_var, conf_min,
    ], axis=1).astype(np.float32)

    # Label: head-0 argmax vs verifier argmax at ts (vpred is length T-1 over positions 1..T-1)
    # vpred index j corresponds to hidden position j+1; argmax @ position ts uses vpred[ts-1+1]?
    # Parent uses: vpred = argmax(h[1:] @ W.T)  -> vpred[i] = next-token from h_{i+1}.
    # Then label = (h0p == vpred[ts]).  vpred is indexed 0..T-2, so vpred[ts] must be valid.
    # Since ts <= valid-1 = SEQ_LEN-3, and vpred has length T-1=SEQ_LEN-1, vpred[ts] is safe.
    label = (h0p == vpred[ts]).astype(np.float32)

    return feat, label, confs0, probs0
