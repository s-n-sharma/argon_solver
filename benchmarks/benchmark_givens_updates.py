"""Benchmark incremental constraint handling with and without Givens rotations.

The script compares three strategies while performing a sequence of row
insertions and deletions on linear systems:

1. ``no_givens_sparse``: repeatedly refactorise with sparse QR via ``sparseqr``.
2. ``givens_sparse``: simulate sparse QR updates by densifying and applying
   Givens-based rank-one updates (acts as an optimistic upper bound).
3. ``givens_dense``: dense QR with true Givens updates using SciPy.

Each strategy is timed over the same update pattern so the relative costs of
refactorisation vs incremental updates become visible.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Iterable, List, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse as sp

try:
    import sparseqr
except ImportError:  # pragma: no cover - optional dependency
    sparseqr = None  # type: ignore


@dataclasses.dataclass
class UpdateBenchmark:
    method: str
    m: int
    n: int
    updates: int
    time_ms: float


def _initial_problem(m: int, n: int, density: float, seed: int = 0) -> sp.coo_matrix:
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=density, format="csr", random_state=rng)
    A.data = rng.standard_normal(A.nnz)
    return A.tocoo()


def _new_row(n: int, density: float, seed: int) -> sp.coo_matrix:
    rng = np.random.default_rng(seed)
    nnz = max(1, int(density * n))
    cols = rng.choice(n, nnz, replace=False)
    data = rng.standard_normal(nnz)
    row = sp.coo_matrix((data, (np.zeros_like(cols), cols)), shape=(1, n))
    return row


def _sequence(num_ops: int, add_first: bool = True) -> List[str]:
    ops: List[str] = []
    toggle = add_first
    for _ in range(num_ops):
        ops.append("add" if toggle else "delete")
        toggle = not toggle
    return ops


def _no_givens_sparse(A0: sp.coo_matrix, ops: Iterable[str], density: float) -> Tuple[float, sp.coo_matrix]:
    if sparseqr is None:
        raise RuntimeError("sparseqr not installed; cannot run sparse benchmarks")
    A = A0.tocsr()
    rng = np.random.default_rng(123)
    start = time.perf_counter()
    for op in ops:
        if op == "add":
            new_row = _new_row(A.shape[1], density, seed=rng.integers(1 << 30))
            A = sp.vstack([A, new_row], format="csr")
        else:
            if A.shape[0] > 1:
                A = A[:-1]
        sparseqr.qr(A, tolerance=1e-12, economy=False)
    elapsed = (time.perf_counter() - start) * 1e3
    return elapsed, A.tocoo()


def _givens_dense(A0: sp.coo_matrix, ops: Iterable[str], density: float) -> Tuple[float, np.ndarray]:
    A = A0.toarray()
    rng = np.random.default_rng(456)
    q, r = scipy.linalg.qr(A, mode="economic")
    start = time.perf_counter()
    for op in ops:
        if op == "add":
            new_row = _new_row(A.shape[1], density, seed=rng.integers(1 << 30)).toarray().ravel()
            q, r = scipy.linalg.qr_insert(q, r, new_row, k=r.shape[0])
        else:
            if r.shape[0] > 1:
                q, r = scipy.linalg.qr_delete(q, r, r.shape[0] - 1)
    elapsed = (time.perf_counter() - start) * 1e3
    return elapsed, r


def _givens_sparse_proxy(A0: sp.coo_matrix, ops: Iterable[str], density: float) -> Tuple[float, sp.coo_matrix]:
    """Proxy sparse Givens by maintaining CSR but updating factors densely.

    This approximates the cost of an ideal sparse-Givens implementation and
    acts as an upper bound for what a dedicated implementation might deliver.
    """
    if sparseqr is None:
        raise RuntimeError("sparseqr not installed; cannot run sparse benchmarks")
    A = A0.tocsr()
    rng = np.random.default_rng(789)
    q, r = scipy.linalg.qr(A.toarray(), mode="economic")
    start = time.perf_counter()
    for op in ops:
        if op == "add":
            new_row = _new_row(A.shape[1], density, seed=rng.integers(1 << 30))
            A = sp.vstack([A, new_row], format="csr")
            q, r = scipy.linalg.qr_insert(q, r, new_row.toarray().ravel(), k=r.shape[0])
        else:
            if A.shape[0] > 1:
                A = A[:-1]
                if r.shape[0] > 1:
                    q, r = scipy.linalg.qr_delete(q, r, r.shape[0] - 1)
    elapsed = (time.perf_counter() - start) * 1e3
    return elapsed, A.tocoo()


def run_update_benchmark(m: int = 500, n: int = 100, num_ops: int = 10, density: float = 0.01) -> List[UpdateBenchmark]:
    A0 = _initial_problem(m, n, density=density)
    ops = _sequence(num_ops)

    timings: List[UpdateBenchmark] = []

    sparse_time, _ = _no_givens_sparse(A0, ops, density)
    timings.append(UpdateBenchmark("no_givens_sparse", m, n, num_ops, sparse_time))

    sparse_givens_time, _ = _givens_sparse_proxy(A0, ops, density)
    timings.append(UpdateBenchmark("givens_sparse_proxy", m, n, num_ops, sparse_givens_time))

    dense_time, _ = _givens_dense(A0, ops, density)
    timings.append(UpdateBenchmark("givens_dense", m, n, num_ops, dense_time))

    return timings


def _format_timings(rows: Iterable[UpdateBenchmark]) -> str:
    header = "method,m,n,updates,time_ms"
    lines = [header]
    for row in rows:
        lines.append(f"{row.method},{row.m},{row.n},{row.updates},{row.time_ms:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_update_benchmark()
    print(_format_timings(results))
