"""QB factorisation helpers mirroring the Parla interfaces."""

from __future__ import annotations

import numpy as np

from .linalg_wrappers import matmul


class QBDecomposer:
    def __init__(self, range_finder) -> None:
        self.range_finder = range_finder

    def factorize(self, matrix, rank: int):
        q = self.range_finder(matrix, rank)
        b = matmul(q.T, matrix)
        return q, b


class QB2(QBDecomposer):
    """Alias for the two-pass QB routine used in Parla."""

    def factorize(self, matrix, rank: int):
        q, b = super().factorize(matrix, rank)
        # ensure B is shaped consistently even when rank exceeds dimensions
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return q, b
