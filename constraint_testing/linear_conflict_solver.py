from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .linear_conflict import LinearSystemModel, IterativeLinearizer, analyze_residuals, ConflictAnalysis


@dataclass
class LinearConflictResult:
    """Summary of the iterative linear consistency check."""

    is_consistent: bool
    iterate: np.ndarray
    residual: np.ndarray
    residual_norm: float
    iterations: int
    conflict_analysis: ConflictAnalysis
    conflicting_indices: Sequence[int]
    iis_verified: bool


class LinearConflictSolver:
    """Conflict detector for linear systems using LSQR or QR iterations."""

    def __init__(
        self,
        solver: str = "lsqr",
        tol: float = 1e-8,
        step_tol: float = 1e-10,
        max_iters: int = 20,
        damping: float = 1e-9,
    ) -> None:
        self.linearizer = IterativeLinearizer(
            solver=solver, tol=tol, step_tol=step_tol, max_iters=max_iters, damping=damping
        )
        self.tol = float(tol)

    def analyze(
        self,
        A: sparse.spmatrix | np.ndarray,
        b: Iterable[float],
        initial_guess: np.ndarray | None = None,
        verify_iis: bool = False,
    ) -> LinearConflictResult:
        """Run the iterative conflict detector on Ax = b."""
        A_op = self._ensure_matrix(A)
        b_vec = np.asarray(b, dtype=float).reshape(-1)
        n_vars = A_op.shape[1]
        if initial_guess is None:
            initial_guess = np.zeros(n_vars, dtype=float)

        model = LinearSystemModel(matrix=A_op, rhs=b_vec, initial_solution=initial_guess)
        linearization = self.linearizer.run(model)
        analysis = analyze_residuals(linearization.residual, self.tol)
        conflict_indices = tuple(int(idx) for idx in analysis.conflict_indices)
        iis_verified = False
        if verify_iis and conflict_indices:
            iis_verified = self._verify_iis(A_op, b_vec, conflict_indices)
        is_consistent = analysis.residual_norm <= self.tol
        return LinearConflictResult(
            is_consistent=is_consistent,
            iterate=linearization.iterate,
            residual=linearization.residual,
            residual_norm=analysis.residual_norm,
            iterations=linearization.iterations,
            conflict_analysis=analysis,
            conflicting_indices=conflict_indices,
            iis_verified=iis_verified,
        )

    @staticmethod
    def _ensure_matrix(A: sparse.spmatrix | np.ndarray) -> sparse.spmatrix | np.ndarray:
        if sparse.isspmatrix(A):
            return A.tocsr()
        return np.asarray(A, dtype=float)

    def _verify_iis(
        self,
        A: sparse.spmatrix | np.ndarray,
        b: np.ndarray,
        indices: Sequence[int],
    ) -> bool:
        index_array = np.array(sorted(set(indices)), dtype=int)
        if index_array.size == 0:
            return False
        if not self._is_inconsistent(A, b, index_array):
            return False
        for idx in index_array:
            subset = index_array[index_array != idx]
            if subset.size == 0:
                continue
            if not self._is_consistent(A, b, subset):
                return False
        return True

    def _is_consistent(self, A, b: np.ndarray, indices: np.ndarray) -> bool:
        residual = self._subset_residual(A, b, indices)
        return residual <= self.tol

    def _is_inconsistent(self, A, b: np.ndarray, indices: np.ndarray) -> bool:
        residual = self._subset_residual(A, b, indices)
        return residual > self.tol

    def _subset_residual(self, A, b: np.ndarray, indices: np.ndarray) -> float:
        A_sub = A[indices, :]
        b_sub = b[indices]
        if sparse.isspmatrix(A_sub):
            iter_lim = max(1, A_sub.shape[1]) * 10
            result = lsqr(A_sub, b_sub, atol=self.tol, btol=self.tol, damp=self.linearizer.damping, iter_lim=iter_lim)
            x = result[0]
            residual = A_sub @ x - b_sub
        else:
            A_sub = np.asarray(A_sub, dtype=float)
            x, *_ = np.linalg.lstsq(A_sub, b_sub, rcond=self.linearizer.damping)
            residual = A_sub @ x - b_sub
        return float(np.linalg.norm(residual))
