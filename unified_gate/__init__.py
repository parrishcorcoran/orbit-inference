"""unified-gate: confidence-gated adaptive LLM inference on a boundary manifold."""
from __future__ import annotations

from .features import extract_all_features, FEATURE_NAMES, N_FEATURES
from .gate import Gate

__all__ = ["extract_all_features", "FEATURE_NAMES", "N_FEATURES", "Gate"]
__version__ = "0.1.0"
