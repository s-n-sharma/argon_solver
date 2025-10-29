"""Benchmark SuiteSparse-based sparse QR conflict analysis against dense QR.

This script sweeps multiple matrix generators and problem sizes, introduces
controlled inconsistencies, and times how quickly inconsistent rows are
identified by:

1. The sparse QR route via the ``sparseqr`` Python bindings (SuiteSparse).
2. A dense rank-revealing QR using SciPy's LAPACK wrapper (pivoted QR).

It prints a summary table for each generator with timing and accuracy stats.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse as sp

try:
    import sparseqr
except ImportError:  # pragma: no cover - optional dependency
    sparseqr = None  # type: ignore


@dataclasses.dataclass
class BenchmarkResult:
    generator: str
    size: int
    nnz: int
    sparse_time_ms: float
    dense_time_ms: float
    sparse_detected: bool
    dense_detected: bool


def _make_inconsistent_rhs(A: sp.coo_matrix, noise_scale: float = 1.0) -> np.ndarray:
    """Construct a right-hand side that guarantees inconsistency."""
    rng = np.random.default_rng()
    # Start with a consistent rhs using a random ground-truth x.
    x_true = rng.standard_normal(A.shape[1])
    b = A @ x_true
    # Corrupt the final row heavily to guarantee infeasibility.
    b = np.asarray(b).reshape(-1)
    b[-1] += noise_scale * (10.0 + abs(b[-1]))
    return b


def _random_sparse_matrix(m: int, n: int, density: float, seed: int) -> sp.coo_matrix:
    rng = np.random.default_rng(seed)
    matrix = sp.random(m, n, density=density, format="coo", random_state=rng, dtype=float)
    matrix.data = rng.standard_normal(matrix.nnz)
    return matrix


def _banded_circular_graph(m: int, n: int) -> sp.coo_matrix:
    """Build constraints resembling a circular factor graph."""
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i in range(m):
        rows.append(i)
        cols.append(i % n)
        data.append(1.0)
        rows.append(i)
        cols.append((i + 1) % n)
        data.append(-1.0)
    return sp.coo_matrix((data, (rows, cols)), shape=(m, n))


def _block_sparse_diagonal(m: int, n: int, block: int) -> sp.coo_matrix:
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    val = 1.0
    for r in range(m):
        c = (r // block) % n
        rows.append(r)
        cols.append(c)
        data.append(val)
        val = -val
    return sp.coo_matrix((data, (rows, cols)), shape=(m, n))


MatrixGenerator = Callable[[int, int, int], sp.coo_matrix]


def _sparse_conflict_detection(A: sp.coo_matrix, b: np.ndarray) -> Tuple[float, bool]:
    if sparseqr is None:
        raise RuntimeError("sparseqr Python bindings not available")
    start = time.perf_counter()
    Q, R, E, rank = sparseqr.qr(A, tolerance=1e-12, economy=False)
    c = Q.transpose() @ b.reshape(-1, 1)
    residual_norm = np.linalg.norm(c[rank:])
    elapsed = (time.perf_counter() - start) * 1e3
    detected = bool(residual_norm > 1e-9)
    return elapsed, detected


def _dense_conflict_detection(A: sp.coo_matrix, b: np.ndarray) -> Tuple[float, bool]:
    dense_A = A.toarray()
    start = time.perf_counter()
    q, r, piv = scipy.linalg.qr(dense_A, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(r)
    c = q.T @ b
    residual_norm = np.linalg.norm(c[rank:])
    elapsed = (time.perf_counter() - start) * 1e3
    detected = bool(residual_norm > 1e-9)
    return elapsed, detected


def run_conflict_benchmark() -> List[BenchmarkResult]:
    if sparseqr is None:
        raise RuntimeError(
            "sparseqr package not installed. Install via `pip install sparseqr` to run benchmarks."
        )

    sizes = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
    generators: Dict[str, Callable[[int, int, int], sp.coo_matrix]] = {
        "random_sparse": lambda m, n, seed: _random_sparse_matrix(m, n, density=0.01, seed=seed),
        "circular_graph": lambda m, n, seed: _banded_circular_graph(m, n),
        "block_diagonal": lambda m, n, seed: _block_sparse_diagonal(m, n, block=max(1, n // 10)),
    }

    results: List[BenchmarkResult] = []
    for name, generator in generators.items():
        for size in sizes:
            m = size
            n = max(2, size // 2)
            A = generator(m, n, seed=42)
            b = _make_inconsistent_rhs(A)
            sparse_time, sparse_detected = _sparse_conflict_detection(A, b)
            dense_time, dense_detected = _dense_conflict_detection(A, b)
            results.append(
                BenchmarkResult(
                    generator=name,
                    size=size,
                    nnz=A.nnz,
                    sparse_time_ms=sparse_time,
                    dense_time_ms=dense_time,
                    sparse_detected=sparse_detected,
                    dense_detected=dense_detected,
                )
            )
    return results


def _format_results(results: Iterable[BenchmarkResult]) -> str:
    lines = [
        "generator,size,nnz,sparse_ms,dense_ms,sparse_detected,dense_detected",
    ]
    for row in results:
        lines.append(
            f"{row.generator},{row.size},{row.nnz},{row.sparse_time_ms:.3f},{row.dense_time_ms:.3f},"
            f"{int(row.sparse_detected)},{int(row.dense_detected)}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    all_results = run_conflict_benchmark()
    print(_format_results(all_results))
