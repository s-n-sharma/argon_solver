from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import sparse

Array = np.ndarray


@dataclass
class LinearSystemModel:
    """Represent a linear system Ax = b and an initial solution guess."""

    matrix: Union[Array, sparse.spmatrix]
    rhs: Array
    initial_solution: Array

    def __post_init__(self) -> None:
        if sparse.isspmatrix(self.matrix):
            self.matrix = self.matrix.tocsr()
        else:
            self.matrix = np.asarray(self.matrix, dtype=float)
        if getattr(self.matrix, "ndim", 2) != 2:
            raise ValueError("Matrix must be two-dimensional")
        self.rhs = np.asarray(self.rhs, dtype=float).reshape(-1)
        self.initial_solution = np.asarray(self.initial_solution, dtype=float)
        m, n = self._shape
        if self.initial_solution.shape[0] != n:
            raise ValueError("Initial solution dimension does not match matrix column count")
        if self.rhs.shape[0] != m:
            raise ValueError("Right-hand side dimension does not match matrix row count")

    @property
    def _shape(self) -> tuple[int, int]:
        if sparse.isspmatrix(self.matrix):
            return self.matrix.shape
        return self.matrix.shape

    def residual(self, x: Array) -> Array:
        return self._matvec(x) - self.rhs

    def jacobian(self) -> Union[Array, sparse.spmatrix]:
        return self.matrix

    def _matvec(self, x: Array) -> Array:
        if sparse.isspmatrix(self.matrix):
            return self.matrix @ x
        return self.matrix @ x
