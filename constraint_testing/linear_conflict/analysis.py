from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ConflictAnalysis:
    residual_norm: float
    conflict_indices: Sequence[int]
    threshold: float


def analyze_residuals(residual: np.ndarray, tol: float) -> ConflictAnalysis:
    abs_residual = np.abs(residual)
    sorted_indices = np.argsort(abs_residual)[::-1]
    significant = sorted_indices[abs_residual[sorted_indices] > tol]
    return ConflictAnalysis(
        residual_norm=float(np.linalg.norm(residual)),
        conflict_indices=tuple(int(idx) for idx in significant),
        threshold=float(tol),
    )
