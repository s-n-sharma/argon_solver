"""Sketch-based solvers for overdetermined least-squares problems.

The implementations in this module are inspired by the ``parla`` project,
particularly the classes defined in ``parla.drivers``.  The focus here is on
providing lightweight, dependency-free solvers that capture the same flow:

1. Construct an oblivious sketch of the input matrix.
2. Factor the sketched system to obtain a preconditioner or solution guess.
3. Optionally refine the solution with an iterative method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as spla

from .sketchers import SketchSample, make_sketch, apply_sketch


@dataclass
class SketchSolverOptions:
    """Configuration for sketch-based least squares solvers."""

    mode: str = "precondition"  # ``"solve"`` or ``"precondition"``
    sketch_method: str = "sparse_sign"
    sampling_factor: float = 3.0
    sketch_size: Optional[int] = None
    sparsity: int = 8
    regularization: float = 0.0
    rank_tol: float = 1e-12
    lsqr_tol: float = 1e-8
    lsqr_iter_lim: Optional[int] = None
    random_state: Optional[int] = None
    warm_start: bool = True


@dataclass
class SolverLog:
    """Runtime diagnostics collected during a solver call."""

    time_sketch: float = 0.0
    time_factor: float = 0.0
    time_iterate: float = 0.0
    iterations: int = 0
    lsqr_flag: Optional[int] = None
    warm_start_residual: Optional[float] = None


@dataclass
class SolverOutput:
    """Container returned by sketch solvers."""

    x: np.ndarray
    sample: SketchSample
    sketch_size: int
    rank_estimate: int
    log: SolverLog


def _ensure_csr(matrix: spa.spmatrix | np.ndarray) -> spa.csr_matrix:
    if spa.issparse(matrix):
        return matrix.tocsr()
    return spa.csr_matrix(matrix)


def _determine_embedding_dim(options: SketchSolverOptions, n_rows: int, n_cols: int) -> int:
    if options.sketch_size is not None:
        d = int(options.sketch_size)
    else:
        d = int(np.ceil(options.sampling_factor * n_cols))
    d = max(n_cols, min(n_rows, d))
    return d


class SketchAndSolveLSQ:
    """One-shot sketch-and-solve solver similar to ``parla``'s ``SSO1``."""

    def __init__(self, options: SketchSolverOptions) -> None:
        self.options = options

    def __call__(self, A: spa.spmatrix | np.ndarray, b: np.ndarray) -> SolverOutput:
        A_csr = _ensure_csr(A)
        b_vec = np.asarray(b, dtype=float).reshape(-1)
        m, n = A_csr.shape
        if b_vec.shape[0] != m:
            raise ValueError("Shapes of A and b do not align.")

        d = _determine_embedding_dim(self.options, m, n)
        rng = np.random.default_rng(self.options.random_state)

        log = SolverLog()
        tic = time.perf_counter()
        sample = make_sketch(
            method=self.options.sketch_method,
            embedding_dim=d,
            ambient_dim=m,
            rng=rng,
            sparsity=self.options.sparsity,
        )
        log.time_sketch = time.perf_counter() - tic

        A_ske = apply_sketch(sample, A_csr)
        b_ske = np.asarray(apply_sketch(sample, b_vec), dtype=float).reshape(-1)
        if self.options.regularization > 0:
            sqrt_delta = np.sqrt(self.options.regularization)
            A_ske = spa.vstack([
                A_ske,
                sqrt_delta * spa.identity(n, format="csc"),
            ])
            b_ske = np.concatenate([b_ske, np.zeros(n)])

        tic = time.perf_counter()
        A_dense = A_ske.toarray()
        lstsq_res = la.lstsq(A_dense, b_ske, lapack_driver="gelsd")
        x = lstsq_res[0]
        singular_values = lstsq_res[3]
        rank_est = int(np.sum(singular_values > self.options.rank_tol))
        log.time_factor = time.perf_counter() - tic

        return SolverOutput(
            x=x,
            sample=sample,
            sketch_size=d,
            rank_estimate=rank_est,
            log=log,
        )


def _build_preconditioned_operator(A: spa.csr_matrix, R: np.ndarray) -> spla.LinearOperator:
    m, n = A.shape

    def matvec(z: np.ndarray) -> np.ndarray:
        y = la.solve_triangular(R, z, lower=False)
        return A @ y

    def rmatvec(w: np.ndarray) -> np.ndarray:
        v = A.T @ w
        return la.solve_triangular(R.T, v, lower=True)

    return spla.LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)


class SketchAndPreconditionLSQ:
    """Sketch-and-precondition solver reminiscent of ``parla``'s ``SPO``."""

    def __init__(self, options: SketchSolverOptions) -> None:
        self.options = options

    def __call__(self, A: spa.spmatrix | np.ndarray, b: np.ndarray) -> SolverOutput:
        A_csr = _ensure_csr(A)
        b_vec = np.asarray(b, dtype=float).reshape(-1)
        m, n = A_csr.shape
        if b_vec.shape[0] != m:
            raise ValueError("Shapes of A and b do not align.")

        d = _determine_embedding_dim(self.options, m, n)
        rng = np.random.default_rng(self.options.random_state)

        log = SolverLog()
        tic = time.perf_counter()
        sample = make_sketch(
            method=self.options.sketch_method,
            embedding_dim=d,
            ambient_dim=m,
            rng=rng,
            sparsity=self.options.sparsity,
        )
        log.time_sketch = time.perf_counter() - tic

        tic = time.perf_counter()
        A_ske = apply_sketch(sample, A_csr)
        if self.options.regularization > 0:
            sqrt_delta = np.sqrt(self.options.regularization)
            A_ske = spa.vstack([
                A_ske,
                sqrt_delta * spa.identity(n, format="csc"),
            ])
        A_dense = A_ske.toarray()
        use_qr = True
        try:
            Q, R = la.qr(A_dense, mode="economic")
            diag = np.abs(np.diag(R)) if R.ndim == 2 else np.empty(0)
            rank_est = int(np.sum(diag > self.options.rank_tol))
            if rank_est < n:
                raise la.LinAlgError("rank deficient sketch")
        except la.LinAlgError:
            use_qr = False
            gram = A_dense.T @ A_dense
            ridge = max(self.options.regularization, self.options.rank_tol)
            if ridge <= 0:
                ridge = 1e-10
            gram += ridge * np.eye(n)
            R = la.cholesky(gram, lower=False)
            rank_est = int(np.sum(np.abs(np.diag(R)) > self.options.rank_tol))
            Q = None
        log.time_factor = time.perf_counter() - tic

        b_ske = np.asarray(apply_sketch(sample, b_vec), dtype=float).reshape(-1)
        if self.options.regularization > 0:
            b_ske = np.concatenate([b_ske, np.zeros(n)])

        x0 = None
        if self.options.warm_start:
            if use_qr and Q is not None:
                z = Q.T @ b_ske
                tentative = la.solve_triangular(R, z, lower=False)
            else:
                normal_rhs = A_dense.T @ b_ske
                y_tmp = la.solve_triangular(R, normal_rhs, lower=False, trans="T")
                tentative = la.solve_triangular(R, y_tmp, lower=False)
            x0 = tentative
            residual = la.norm(A_csr @ tentative - b_vec)
            log.warm_start_residual = residual

        operator = _build_preconditioned_operator(A_csr, R)
        lsqr_tol = self.options.lsqr_tol
        iter_lim = self.options.lsqr_iter_lim or (2 * n)

        tic = time.perf_counter()
        lsqr_res = spla.lsqr(
            operator,
            b_vec,
            atol=lsqr_tol,
            btol=lsqr_tol,
            iter_lim=iter_lim,
            x0=None if x0 is None else R @ x0,
        )
        log.time_iterate = time.perf_counter() - tic
        log.iterations = lsqr_res[2]
        log.lsqr_flag = lsqr_res[1]

        y = lsqr_res[0]
        x = la.solve_triangular(R, y, lower=False)

        return SolverOutput(
            x=x,
            sample=sample,
            sketch_size=d,
            rank_estimate=rank_est,
            log=log,
        )


def solve_least_squares(
    A: spa.spmatrix | np.ndarray,
    b: np.ndarray,
    options: SketchSolverOptions,
) -> SolverOutput:
    """Solve a least squares problem with the requested sketching mode."""

    mode = options.mode.lower()
    if mode == "solve":
        solver = SketchAndSolveLSQ(options)
    elif mode in {"precondition", "fossils", "precond"}:
        solver = SketchAndPreconditionLSQ(options)
    else:
        raise ValueError(f"Unknown solver mode '{options.mode}'.")
    return solver(A, b)


__all__ = [
    "SketchSolverOptions",
    "SolverLog",
    "SolverOutput",
    "SketchAndSolveLSQ",
    "SketchAndPreconditionLSQ",
    "solve_least_squares",
]
