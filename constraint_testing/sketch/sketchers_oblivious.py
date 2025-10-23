"""Oblivious sketch constructions used by the randomized SVD utility."""

from __future__ import annotations

import numpy as np


def gaussian(n_rows: int, n_cols: int, random_state: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size=(n_rows, n_cols))
