"""Randomized SVD solver loosely adapted from Parla's driver implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from .qb import QB2
from .rangefinders import RF1
from .sketchers_aware import RS1
from . import sketchers_oblivious as oblivious
from .linalg_wrappers import matmul, to_dense


@dataclass
class RandomizedSVDResult:
    solution: np.ndarray
    solved_variables: Dict[int, float]
    inconsistent_rows: Sequence[int]
    singular_values: np.ndarray
    rank: int
    residual: np.ndarray


class RandomizedSVDSolver:
    def __init__(
        self,
        target_rank: int | None = None,
        oversample: int = 10,
        power_iterations: int = 1,
        epsilon: float = 1e-9,
        random_state: int | None = None,
    ) -> None:
        self.target_rank = target_rank
        self.oversample = oversample
        self.power_iterations = power_iterations
        self.epsilon = float(epsilon)
        self.random_state = random_state
        self._aware_sketch = RS1(random_state=random_state)
        self._range_finder = RF1(
            oversample=oversample,
            n_iter=power_iterations,
            random_state=random_state,
        )
        self._qb = QB2(self._range_finder)

    def solve(self, A, b: Iterable[float]) -> RandomizedSVDResult:
        A_dense = to_dense(A)
        b_vec = np.asarray(b, dtype=float).reshape(-1)
        m, n = A_dense.shape
        if m == 0 or n == 0:
            return RandomizedSVDResult(
                solution=np.zeros(n, dtype=float),
                solved_variables={},
                inconsistent_rows=(),
                singular_values=np.array([], dtype=float),
                rank=0,
                residual=np.zeros(m, dtype=float),
            )

        target_rank = self.target_rank or min(m, n)
        target_rank = min(target_rank, min(m, n))
        if target_rank <= 0:
            target_rank = min(m, n)

        # optional data-aware preprocessing (not strictly necessary but keeps API similar)
        _ = self._aware_sketch(A_dense, min(target_rank + self.oversample, n))

        Q, B = self._qb.factorize(A_dense, target_rank)
        U_tilde, singular_values, Vt = np.linalg.svd(B, full_matrices=False)

        effective_rank = int(np.sum(singular_values > self.epsilon))
        if effective_rank == 0:
            residual = b_vec.copy()
            inconsistent_rows = tuple(np.nonzero(np.abs(residual) > self.epsilon)[0])
            return RandomizedSVDResult(
                solution=np.zeros(n, dtype=float),
                solved_variables={},
                inconsistent_rows=inconsistent_rows,
                singular_values=singular_values,
                rank=0,
                residual=residual,
            )

        U = matmul(Q, U_tilde[:, :effective_rank])
        Vt_trunc = Vt[:effective_rank, :]
        s_trunc = singular_values[:effective_rank]

        uTb = U.T @ b_vec
        coeffs = uTb[:effective_rank] / s_trunc
        solution = Vt_trunc.T @ coeffs

        solved_variables: Dict[int, float] = {}
        for i in range(n):
            col = Vt_trunc[:, i]
            coeff_norm = float(np.dot(col, col))
            if abs(coeff_norm - 1.0) <= self.epsilon:
                solved_variables[i] = float(np.round(solution[i]))

        residual = b_vec - A_dense @ solution
        row_norms = np.linalg.norm(A_dense, axis=1)
        inconsistent_rows = [
            idx
            for idx, (row_norm, resid_val) in enumerate(zip(row_norms, residual))
            if row_norm <= self.epsilon and abs(resid_val) > self.epsilon
        ]

        return RandomizedSVDResult(
            solution=solution,
            solved_variables=solved_variables,
            inconsistent_rows=tuple(inconsistent_rows),
            singular_values=singular_values,
            rank=effective_rank,
            residual=residual,
        )

    def sketch_matrix(self, A):
        A_dense = to_dense(A)
        sketch_size = min(A_dense.shape)
        sketch = oblivious.gaussian(A_dense.shape[1], sketch_size, random_state=self.random_state)
        return matmul(A_dense, sketch)
