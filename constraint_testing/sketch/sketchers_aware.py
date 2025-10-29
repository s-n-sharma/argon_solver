"""Simple data-aware sketchers used by the randomized SVD pipeline."""

from __future__ import annotations

import numpy as np

from .linalg_wrappers import to_dense


class RS1:
    """Column sampling sketch weighted by column norms."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def __call__(self, matrix, sketch_size: int) -> np.ndarray:
        dense = to_dense(matrix)
        n_cols = dense.shape[1]
        rng = np.random.default_rng(self.random_state)
        norms = np.linalg.norm(dense, axis=0)
        if not np.any(norms):
            norms = np.ones_like(norms)
        probs = norms / norms.sum()
        indices = rng.choice(n_cols, size=sketch_size, replace=True, p=probs)
        scaling = np.sqrt(sketch_size * probs[indices])
        scaling[scaling == 0.0] = 1.0
        sampled = dense[:, indices] / scaling
        return sampled
