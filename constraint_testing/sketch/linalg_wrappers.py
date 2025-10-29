"""Minimal linear algebra helpers inspired by Parla's linalg wrappers."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la


def to_dense(matrix) -> np.ndarray:
    """Convert sparse/dense inputs to a dense ndarray."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix, dtype=float)


def matmul(left, right):
    return np.matmul(left, right)


def adjoint(matrix):
    return np.conjugate(matrix.T)


def qr_reduced(matrix):
    q, r = np.linalg.qr(matrix, mode="reduced")
    return q, r


def svd_economic(matrix):
    return la.svd(matrix, full_matrices=False)


def norm(matrix):
    return np.linalg.norm(matrix)
