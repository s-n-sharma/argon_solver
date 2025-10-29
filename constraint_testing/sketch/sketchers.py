"""Oblivious sketch operators for randomized numerical linear algebra.

This module implements a subset of the sketching primitives provided by the
`parla.comps.sketchers` package. The focus is on fast-to-apply sketches that
work well with sparse CAD constraint matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import scipy.sparse as spa


@dataclass
class SketchSample:
    """Description of the sketch that was applied to a matrix or vector."""

    method: str
    matrix: spa.spmatrix
    params: Dict[str, float]
    seed: Optional[int]


def _ensure_rng(seed: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _gaussian_sketch(d: int, m: int, rng: np.random.Generator) -> spa.csr_matrix:
    data = rng.standard_normal(size=(d, m)) / np.sqrt(d)
    return spa.csr_matrix(data)


def _sparse_sign_sketch(
    d: int,
    m: int,
    rng: np.random.Generator,
    sparsity: int,
) -> spa.csr_matrix:
    sparsity = max(1, int(sparsity))
    nnz = sparsity * m
    rows = rng.integers(0, d, size=nnz)
    cols = np.repeat(np.arange(m), sparsity)
    signs = rng.choice([-1.0, 1.0], size=nnz)
    data = signs / np.sqrt(sparsity)
    return spa.csr_matrix((data, (rows, cols)), shape=(d, m))


def _count_sketch(
    d: int,
    m: int,
    rng: np.random.Generator,
) -> spa.csr_matrix:
    rows = rng.integers(0, d, size=m)
    cols = np.arange(m)
    signs = rng.choice([-1.0, 1.0], size=m)
    data = signs
    return spa.csr_matrix((data, (rows, cols)), shape=(d, m))


def make_sketch(
    method: str,
    embedding_dim: int,
    ambient_dim: int,
    rng: Optional[int | np.random.Generator] = None,
    sparsity: int = 8,
) -> SketchSample:
    """Construct an oblivious sketching matrix.

    Parameters
    ----------
    method:
        Name of the sketching scheme. Supported values are ``"gaussian"``,
        ``"sparse_sign"``, and ``"count"``.
    embedding_dim:
        Number of rows in the sketching matrix.
    ambient_dim:
        Number of columns in the sketching matrix.
    rng:
        Optional random seed or Generator.
    sparsity:
        Number of non-zeros per column for sparse sketches.
    """

    rng_obj = _ensure_rng(rng)
    key = method.lower()
    if key in {"gaussian", "normal"}:
        matrix = _gaussian_sketch(embedding_dim, ambient_dim, rng_obj)
        params = {"embedding_dim": float(embedding_dim)}
    elif key in {"sparse_sign", "sjlt", "sj"}:
        matrix = _sparse_sign_sketch(embedding_dim, ambient_dim, rng_obj, sparsity)
        params = {"embedding_dim": float(embedding_dim), "sparsity": float(sparsity)}
    elif key in {"count", "countsketch"}:
        matrix = _count_sketch(embedding_dim, ambient_dim, rng_obj)
        params = {"embedding_dim": float(embedding_dim)}
    else:
        raise ValueError(f"Unknown sketch method '{method}'.")

    seed_value = None if isinstance(rng, np.random.Generator) else rng
    return SketchSample(method=key, matrix=matrix.tocsr(), params=params, seed=seed_value)


def apply_sketch(sample: SketchSample, array: np.ndarray | spa.spmatrix) -> np.ndarray | spa.spmatrix:
    """Apply a sketching matrix to a vector or matrix."""

    return sample.matrix @ array


__all__ = ["SketchSample", "make_sketch", "apply_sketch"]
