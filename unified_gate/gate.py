"""Deploy-time gate: loads gate_k20.pt, normalizes inputs, emits skip scores."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class _GateMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Gate:
    """Trained Medusa skip-gate. Loads a checkpoint and scores features."""

    def __init__(self, ckpt_path: str | Path = "gate_k20.pt"):
        g = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.K: int = int(g["K"])
        self.feature_indices = np.asarray(g["feature_indices"], dtype=np.int64)
        self.feature_names: list[str] = list(g["feature_names"])
        self.mu = np.asarray(g["mu"], dtype=np.float32)
        self.sd = np.asarray(g["sd"], dtype=np.float32)
        self.frontier = g.get("frontier", [])
        self.thresholds: dict = dict(g.get("thresholds", {}))
        self.mlp = _GateMLP(self.K, int(g["mlp_hidden"]))
        self.mlp.load_state_dict(g["mlp_state"])
        self.mlp.eval()

    def _select_and_norm(self, features: np.ndarray) -> np.ndarray:
        X = features[:, self.feature_indices].astype(np.float32)
        return (X - self.mu) / (self.sd + 1e-9)

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        """Return skip-probability per token. ``features`` is [T, N_FEATURES_FULL]."""
        Xn = self._select_and_norm(features)
        logits = self.mlp(torch.from_numpy(Xn))
        return torch.sigmoid(logits).numpy()

    def skip_mask(self, features: np.ndarray, fidelity: float = 0.95) -> np.ndarray:
        """Boolean mask: True = safe to accept Medusa draft, False = run verifier.

        Uses calibrated threshold for target fidelity ``λ``.
        """
        tau = self.thresholds.get(fidelity, None)
        if tau is None:
            # Find closest λ
            keys = list(self.thresholds.keys())
            if not keys:
                raise KeyError("no calibrated thresholds in checkpoint")
            tau = self.thresholds[min(keys, key=lambda k: abs(k - fidelity))]
        return self.score(features) >= float(tau)


__all__ = ["Gate"]
