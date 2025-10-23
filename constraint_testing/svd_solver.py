from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import scipy.sparse as sp


@dataclass
class SVDResult:
    solution: np.ndarray
    solved_variables: Dict[int, float]
    inconsistent_rows: Sequence[int]
    reduced_matrix: np.ndarray
    reduced_rhs: np.ndarray
    rank: int
    singular_values: np.ndarray


class SVD_Solver:
    """Dense SVD-based solver mimicking the Rust implementation structure."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = float(epsilon)

    def solve(self, A: sp.spmatrix | np.ndarray, b: Iterable[float]) -> SVDResult:
        A_dense = _to_dense(A)
        b_vec = np.asarray(b, dtype=float).reshape(-1)

        m, n = A_dense.shape
        if n == 0 or m == 0:
            return SVDResult(
                solution=np.zeros(n, dtype=float),
                solved_variables={},
                inconsistent_rows=(),
                reduced_matrix=A_dense,
                reduced_rhs=b_vec,
                rank=0,
                singular_values=np.array([], dtype=float),
            )

        U, s, Vh = np.linalg.svd(A_dense, full_matrices=False)
        rank = int(np.sum(s > self.epsilon))
        if rank == 0:
            return SVDResult(
                solution=np.zeros(n, dtype=float),
                solved_variables={},
                inconsistent_rows=tuple(range(m)),
                reduced_matrix=A_dense,
                reduced_rhs=b_vec,
                rank=0,
                singular_values=s,
            )

        vt_recons = Vh[:rank, :]
        solution, *_ = np.linalg.lstsq(A_dense, b_vec, rcond=self.epsilon)

        solved_variables: Dict[int, float] = {}
        for i in range(n):
            coeff_norm = float(np.dot(vt_recons[:, i], vt_recons[:, i]))
            if abs(coeff_norm - 1.0) <= self.epsilon:
                solved_variables[i] = float(np.round(solution[i]))

        adjusted_rhs = b_vec.copy()
        solved_indices = np.array(sorted(solved_variables), dtype=int)
        if solved_indices.size:
            solved_values = np.array([solved_variables[i] for i in solved_indices], dtype=float)
            adjusted_rhs -= A_dense[:, solved_indices] @ solved_values

        remaining_mask = np.ones(n, dtype=bool)
        if solved_indices.size:
            remaining_mask[solved_indices] = False
        reduced_matrix = A_dense[:, remaining_mask]

        inconsistent_rows = []
        row_norms = np.linalg.norm(reduced_matrix, axis=1)
        for idx, (row_norm, rhs_val) in enumerate(zip(row_norms, adjusted_rhs)):
            if row_norm <= self.epsilon and abs(rhs_val) > self.epsilon:
                inconsistent_rows.append(idx)

        remaining_rows_mask = row_norms > self.epsilon
        reduced_matrix = reduced_matrix[remaining_rows_mask, :]
        reduced_rhs = adjusted_rhs[remaining_rows_mask]

        return SVDResult(
            solution=solution,
            solved_variables=solved_variables,
            inconsistent_rows=tuple(inconsistent_rows),
            reduced_matrix=reduced_matrix,
            reduced_rhs=reduced_rhs,
            rank=rank,
            singular_values=s,
        )


def _to_dense(A: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(A):
        return A.toarray()
    return np.asarray(A, dtype=float)
