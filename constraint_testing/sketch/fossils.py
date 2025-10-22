"""FOSSILS: fast sketch-and-precondition solver for least squares.

This module implements a streamlined version of the FOSSILS algorithm from
"Fast and Backward Stable Solvers for Overdetermined Systems" (Hawkins et al.,
2024). The implementation follows the high-level structure of Algorithm 7 in
the paper while omitting a few optional safeguards so it can be used as a
practical building block for conflict detection in CAD constraint systems.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from numpy.linalg import norm
from scipy.linalg import solve_triangular

from .sketching import SketchSample, apply_sparse_sketch


@dataclass
class FossilsConfig:
    """Configuration parameters for the FOSSILS solver."""

    sketch_size: Optional[int] = None
    embedding_oversample: float = 6.0
    random_state: Optional[int] = None
    sketch_method: str = "clarkson_woodruff"
    sparsity_parameter: Optional[int] = None
    rank_tol: float = 1e-12
    residual_tol: float = 1e-8
    lsqr_atol: float = 1e-10
    lsqr_btol: float = 1e-10
    lsqr_iter_lim: Optional[int] = None
    heavy_ball_iters: int = 25
    heavy_ball_alpha: Optional[float] = None
    heavy_ball_beta: Optional[float] = None
    top_k_conflicts: Optional[int] = 20


@dataclass
class FossilsResult:
    """Outputs returned by :func:`analyze_system_with_fossils`."""

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


def _column_scaling(A: spa.csr_matrix) -> Tuple[spa.csr_matrix, np.ndarray]:
    squared = A.multiply(A)
    norms = np.sqrt(np.asarray(squared.sum(axis=0)).ravel())
    norms = np.maximum(norms, 1e-12)
    scale = 1.0 / norms
    D = spa.diags(scale, format="csc")
    return A @ D, scale


def _qr_preconditioner(SA: spa.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
    SA_dense = SA.toarray()
    Q, R = np.linalg.qr(SA_dense, mode="reduced")
    return Q, R


def _build_preconditioned_operator(
    A: spa.csr_matrix,
    R: np.ndarray,
) -> spla.LinearOperator:
    m, n = A.shape

    def matvec(y: np.ndarray) -> np.ndarray:
        z = solve_triangular(R, y, lower=False)
        return A @ z

    def rmatvec(v: np.ndarray) -> np.ndarray:
        w = A.T @ v
        return solve_triangular(R.T, w, lower=True)

    return spla.LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)


def _heavy_ball(
    A: spa.csr_matrix,
    R: np.ndarray,
    b: np.ndarray,
    y0: np.ndarray,
    iterations: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    if iterations <= 0:
        return y0

    y_prev = y0.copy()
    y_curr = y0.copy()

    def apply_preconditioned_normal(y: np.ndarray) -> np.ndarray:
        z = solve_triangular(R, y, lower=False)
        Az = A @ z
        grad = A.T @ Az
        grad = solve_triangular(R.T, grad, lower=True)
        return grad

    c = solve_triangular(R.T, A.T @ b, lower=True)

    for _ in range(iterations):
        grad = c - apply_preconditioned_normal(y_curr)
        y_next = y_curr + alpha * grad + beta * (y_curr - y_prev)
        y_prev, y_curr = y_curr, y_next

    return y_curr


def _default_heavy_ball_params(R: np.ndarray) -> Tuple[float, float]:
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return 1.0, 0.0
    lam_max = float(np.max(diag) ** 2)
    lam_min = float(np.min(diag) ** 2)
    lam_min = max(lam_min, 1e-12)
    omega = math.sqrt(lam_max * lam_min)
    alpha = 2.0 / (math.sqrt(lam_max) + math.sqrt(lam_min)) ** 2
    beta = ((math.sqrt(lam_max) - math.sqrt(lam_min)) / (math.sqrt(lam_max) + math.sqrt(lam_min))) ** 2
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0 / lam_max
    if not np.isfinite(beta) or beta < 0:
        beta = 0.0
    return alpha, beta


def analyze_system_with_fossils(
    A: spa.spmatrix,
    b: np.ndarray,
    config: Optional[FossilsConfig] = None,
) -> FossilsResult:
    """Approximate ``Ax = b`` using the FOSSILS sketch-and-precondition solver."""

    if config is None:
        config = FossilsConfig()

    A_csr = _ensure_csr(A)
    m, n = A_csr.shape
    b_vec = np.asarray(b).reshape(-1)
    if b_vec.shape[0] != m:
        raise ValueError("Shapes of A and b do not align.")

    start = time.perf_counter()

    A_scaled, column_scale = _column_scaling(A_csr)

    sketch_size = config.sketch_size
    if sketch_size is None:
        oversample = max(config.embedding_oversample, 1.0)
        sketch_size = min(m, int(math.ceil(oversample * n)))
        sketch_size = max(sketch_size, n + 4)
    sketch_size = max(1, sketch_size)

    if sketch_size >= m:
        SA = A_scaled.copy()
        sb = b_vec.copy()
        S_identity = spa.identity(m, format="csc")
        sample = SketchSample(
            method="identity",
            sketch_matrix=S_identity,
            seed=config.random_state,
            params={"sketch_size": m},
        )
    else:
        SA, sb, sample = apply_sparse_sketch(
            A=A_scaled,
            b=b_vec,
            sketch_size=sketch_size,
            method=config.sketch_method,
            seed=config.random_state,
            sparsity_parameter=config.sparsity_parameter,
        )

    Q, R = _qr_preconditioner(SA)
    rank_estimate = int(np.sum(np.abs(np.diag(R)) > config.rank_tol))

    operator = _build_preconditioned_operator(A_scaled, R)
    lsqr_res = spla.lsqr(
        operator,
        b_vec,
        atol=config.lsqr_atol,
        btol=config.lsqr_btol,
        iter_lim=config.lsqr_iter_lim,
        show=False,
    )
    y = lsqr_res[0]
    y = np.asarray(y)

    if config.heavy_ball_iters > 0:
        alpha, beta = config.heavy_ball_alpha, config.heavy_ball_beta
        if alpha is None or beta is None:
            alpha, beta = _default_heavy_ball_params(R)
        y = _heavy_ball(
            A_scaled,
            R,
            b_vec,
            y,
            iterations=config.heavy_ball_iters,
            alpha=alpha,
            beta=beta,
        )

    x_scaled = solve_triangular(R, y, lower=False)
    x_hat = column_scale * x_scaled

    residual = b_vec - A_csr @ x_hat
    residual_norm = norm(residual)
    is_conflicting = residual_norm > config.residual_tol
    is_underconstrained = rank_estimate < n

    sorted_indices = np.argsort(np.abs(residual))[::-1]
    if config.top_k_conflicts is not None:
        sorted_indices = sorted_indices[: config.top_k_conflicts]

    solve_time = time.perf_counter() - start

    return FossilsResult(
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
