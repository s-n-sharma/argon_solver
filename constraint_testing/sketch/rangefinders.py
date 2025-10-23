"""Range finder utilities for randomized low-rank approximations."""

from __future__ import annotations

import numpy as np

from . import sketchers_oblivious as oblivious
from .linalg_wrappers import matmul, qr_reduced, to_dense


class RF1:
    """Gaussian range finder with optional power iterations."""

    def __init__(
        self,
        oversample: int = 10,
        n_iter: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.oversample = oversample
        self.n_iter = n_iter
        self.random_state = random_state

    def __call__(self, matrix, rank: int) -> np.ndarray:
        dense = to_dense(matrix)
        m, n = dense.shape
        sketch_cols = min(m, rank + self.oversample)
        omega = oblivious.gaussian(n, sketch_cols, random_state=self.random_state)
        y = matmul(dense, omega)

        for _ in range(self.n_iter):
            y = matmul(dense, matmul(dense.T, y))

        q, _ = qr_reduced(y)
        return q
