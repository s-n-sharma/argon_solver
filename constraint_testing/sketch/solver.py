"""Sketch-and-solve routines for fast constraint diagnostics."""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as spa
from numpy.linalg import norm

from .sketching import SketchSample, apply_sparse_sketch


@dataclass
class SketchConfig:
    """Configuration for the sketch-and-solve analyzer."""

    sketch_size: Optional[int] = None
    random_state: Optional[int] = None
    sketch_method: str = "sparse_sign"
    sparsity_parameter: Optional[int] = None
    rank_tol: float = 1e-10
    residual_tol: float = 1e-8
    top_k_conflicts: Optional[int] = 20


@dataclass
class SketchResult:
    """Container for outputs returned by ``analyze_system_with_sketch``."""

    x_hat: np.ndarray
    residual: np.ndarray
    sorted_indices: np.ndarray
    is_conflicting: bool
    is_underconstrained: bool
    residual_norm: float
    rank_estimate: int
    solve_time: float
    sketch_size: int
    sample: SketchSample


def _ensure_csr(A: spa.spmatrix) -> spa.csr_matrix:
    if spa.issparse(A):
        return A.tocsr()
    return spa.csr_matrix(A)


def _default_sketch_size(m: int, n: int) -> int:
    if m == 0:
        return 1
    baseline = max(4 * n, n + 20)
    return min(m, baseline)


def analyze_system_with_sketch(
    A: spa.spmatrix,
    b: np.ndarray,
    config: Optional[SketchConfig] = None,
) -> SketchResult:
    """Approximate ``Ax = b`` diagnostics using a row-sampling sketch."""

    if config is None:
        config = SketchConfig()

    A_csr = _ensure_csr(A)
    m, n = A_csr.shape
    b_vec = np.asarray(b).reshape(-1)
    if b_vec.shape[0] != m:
        raise ValueError("Shapes of A and b do not align.")

    sketch_size = config.sketch_size or _default_sketch_size(m, n)
    sketch_size = max(1, sketch_size)

    start = time.perf_counter()

    SA, sb, sample = apply_sparse_sketch(
        A=A_csr,
        b=b_vec,
        sketch_size=sketch_size,
        method=config.sketch_method,
        seed=config.random_state,
        sparsity_parameter=config.sparsity_parameter,
    )

    SA_dense = SA.toarray()
    sb_vec = np.asarray(sb).reshape(-1)

    lstsq_result = np.linalg.lstsq(SA_dense, sb_vec, rcond=config.rank_tol)
    x_hat = lstsq_result[0]
    rank_estimate = int(lstsq_result[2])

    residual = b_vec - A_csr @ x_hat
    residual_norm = norm(residual)
    is_conflicting = residual_norm > config.residual_tol
    is_underconstrained = rank_estimate < n

    sorted_indices = np.argsort(np.abs(residual))[::-1]
    if config.top_k_conflicts is not None:
        sorted_indices = sorted_indices[: config.top_k_conflicts]

    solve_time = time.perf_counter() - start

    return SketchResult(
        x_hat=x_hat,
        residual=residual,
        sorted_indices=sorted_indices,
        is_conflicting=is_conflicting,
        is_underconstrained=is_underconstrained,
        residual_norm=residual_norm,
        rank_estimate=rank_estimate,
        solve_time=solve_time,
        sketch_size=sketch_size,
        sample=sample,
    )
