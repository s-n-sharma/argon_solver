from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .models import LinearSystemModel


SolverKind = Literal["lsqr", "qr"]


@dataclass
class LinearizationResult:
    iterate: np.ndarray
    residual: np.ndarray
    converged: bool
    iterations: int
    history: List[float] = field(default_factory=list)


class   IterativeLinearizer:
    """Damped least-squares iterator backed by LSQR or dense QR solves."""

    def __init__(
        self,
        solver: SolverKind = "lsqr",
        tol: float = 1e-8,
        step_tol: float = 1e-10,
        max_iters: int = 20,
        damping: float = 1e-9,
    ) -> None:
        if solver not in ("lsqr", "qr"):
            raise ValueError("solver must be 'lsqr' or 'qr'")
        self.solver = solver
        self.tol = float(tol)
        self.step_tol = float(step_tol)
        self.max_iters = int(max_iters)
        self.damping = float(damping)

    def run(self, model: LinearSystemModel) -> LinearizationResult:
        x = model.initial_solution.copy()
        history: List[float] = []

        for iteration in range(1, self.max_iters + 1):
            residual = model.residual(x)
            residual_norm = norm(residual)
            history.append(residual_norm)
            if residual_norm <= self.tol:
                return LinearizationResult(x, residual, True, iteration, history)

            J = model.jacobian()
            delta = self._solve_linear_system(J, -residual)

            if norm(delta) <= self.step_tol:
                return LinearizationResult(x, residual, residual_norm <= self.tol, iteration, history)

            next_x = x + delta
            next_residual = model.residual(next_x)
            if norm(next_residual) > residual_norm:
                # simple backtracking to maintain descent
                next_x = x + 0.5 * delta
                next_residual = model.residual(next_x)

            x = next_x

        final_residual = model.residual(x)
        return LinearizationResult(x, final_residual, False, self.max_iters, history)

    def _solve_linear_system(self, J, rhs: np.ndarray) -> np.ndarray:
        if sparse.isspmatrix(J):
            J_op = J
        else:
            J_op = np.asarray(J, dtype=float)
        if self.solver == "lsqr":
            result = lsqr(J_op, rhs, atol=self.tol, btol=self.tol, damp=self.damping, iter_lim=10 * J_op.shape[1])
            delta = result[0]
        else:
            if sparse.isspmatrix(J_op):
                J_dense = J_op.toarray()
            else:
                J_dense = np.asarray(J_op, dtype=float)
            delta, *_ = np.linalg.lstsq(J_dense, rhs, rcond=self.damping)
        return np.asarray(delta, dtype=float)
