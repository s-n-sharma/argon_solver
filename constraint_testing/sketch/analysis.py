"""High-level constraint diagnostics built on sketch-based solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as spa

from .sketchers import SketchSample
from .solvers import (
    SketchAndPreconditionLSQ,
    SketchAndSolveLSQ,
    SketchSolverOptions,
    SolverLog,
    SolverOutput,
)


@dataclass
class AnalysisConfig:
    """Controls classification thresholds for constraint diagnostics."""

    residual_tol_rel: float = 1e-7
    residual_tol_abs: float = 1e-8
    top_k_conflicts: Optional[int] = 20


@dataclass
class AnalysisResult:
    """Summary statistics returned by :func:`analyze_linear_system`."""

    x_hat: np.ndarray
    residual: np.ndarray
    residual_norm: float
    residual_rel: float
    rank_estimate: int
    is_consistent: bool
    is_underconstrained: bool
    conflicting_indices: np.ndarray
    sample: SketchSample
    sketch_size: int
    solver_log: SolverLog


def _ensure_csr(A: spa.spmatrix | np.ndarray) -> spa.csr_matrix:
    if spa.issparse(A):
        return A.tocsr()
    return spa.csr_matrix(A)


def _select_solver(options: SketchSolverOptions):
    mode = options.mode.lower()
    if mode == "solve":
        return SketchAndSolveLSQ(options)
    if mode in {"precondition", "fossils", "precond"}:
        return SketchAndPreconditionLSQ(options)
    raise ValueError(f"Unknown solver mode '{options.mode}'.")


def analyze_linear_system(
    A: spa.spmatrix | np.ndarray,
    b: np.ndarray,
    solver_options: Optional[SketchSolverOptions] = None,
    analysis_config: Optional[AnalysisConfig] = None,
) -> AnalysisResult:
    """Diagnose consistency and rank properties of ``Ax = b`` using sketches."""

    options = solver_options or SketchSolverOptions()
    settings = analysis_config or AnalysisConfig()

    A_csr = _ensure_csr(A)
    b_vec = np.asarray(b, dtype=float).reshape(-1)
    if b_vec.shape[0] != A_csr.shape[0]:
        raise ValueError("Shapes of A and b do not align.")

    solver = _select_solver(options)
    output: SolverOutput = solver(A_csr, b_vec)

    x_hat = output.x
    residual = A_csr @ x_hat - b_vec
    residual_norm = float(np.linalg.norm(residual))
    denom = max(float(np.linalg.norm(b_vec)), 1.0)
    residual_rel = residual_norm / denom

    is_consistent = residual_norm <= settings.residual_tol_abs or residual_rel <= settings.residual_tol_rel
    n_cols = A_csr.shape[1]
    is_underconstrained = output.rank_estimate < n_cols

    if settings.top_k_conflicts is None or settings.top_k_conflicts <= 0:
        conflicting = np.array([], dtype=int)
    else:
        order = np.argsort(np.abs(residual))[::-1]
        conflicting = order[: settings.top_k_conflicts]

    return AnalysisResult(
        x_hat=x_hat,
        residual=residual,
        residual_norm=residual_norm,
        residual_rel=residual_rel,
        rank_estimate=output.rank_estimate,
        is_consistent=is_consistent,
        is_underconstrained=is_underconstrained,
        conflicting_indices=conflicting,
        sample=output.sample,
        sketch_size=output.sketch_size,
        solver_log=output.log,
    )


__all__ = ["AnalysisConfig", "AnalysisResult", "analyze_linear_system"]
