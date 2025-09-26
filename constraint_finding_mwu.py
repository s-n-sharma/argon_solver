import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog
import time
from typing import List, Tuple, Optional, Callable, Dict
import pandas as pd



# ---------------- Core LP Helpers (same as baseline) ---------------- #

def l1_dual_via_linprog(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solve the L1 LP once and return equality duals via HiGHS (reference method)."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    I = np.eye(m)
    A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n
    bounds_r = [(0, None)] * (2 * m)
    bounds = bounds_x + bounds_r
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs')
    if not res.success:
        return None
    try:
        dual_values = np.asarray(res.eqlin.marginals, dtype=float)
    except AttributeError:
        dual_values = np.asarray(res['eqlin']['marginals'], dtype=float)
    return dual_values

# Convenience baseline finder (same behavior as L1_find in your file)
def L1_find_via_linprog(A: np.ndarray, b: np.ndarray, rel_threshold: float = 1 - 3e-2) -> List[int]:
    duals = l1_dual_via_linprog(A, b)
    if duals is None or duals.size == 0:
        return []
    max_abs = float(np.max(np.abs(duals)))
    if max_abs == 0:
        return []
    mask = np.abs(duals) >= rel_threshold * max_abs
    return [i for i, flag in enumerate(mask) if flag]

# ---------------- MWU-inspired Dual Solver ---------------- #

def _thin_colspace_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return a thin orthonormal basis Q for col(A) using SVD with rank detection.
    Projection onto col(A):    P_col y = Q @ (Q.T @ y)
    Projection onto N(A^T):    P_null y = y - Q @ (Q.T @ y)
    """
    # SVD-based basis is robust under rank deficiency
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((A.shape[0], 0))
    smax = s[0]
    r = int(np.sum(s > max(tol, 1e-12) * smax))
    if r == 0:
        return np.zeros((A.shape[0], 0))
    return U[:, :r]


def mwu_dual_solve(
    A: np.ndarray,
    b: np.ndarray,
    steps: int = 120,
    eta: Optional[float] = None,
    reproject_every: int = 0,
    verbose: bool = False,
    feas_tol: float = 1e-6,
) -> np.ndarray:
    """Compute dual y in [-1,1] s.t. A^T y ≈ 0 maximizing b^T y via fast projected ascent.

    Uses a precomputed thin column-space basis Q of A, so projection onto N(A^T)
    is y <- y - Q(Q^T y), costing O(m n) per iteration instead of solving least squares.

    Parameters:
      - steps: max number of iterations
      - eta: step size; if None, uses 0.9 / (||b||_2 + 1e-9)
      - reproject_every: unused in fast projector; kept for API compatibility
      - feas_tol: early stop when ||A^T y||_inf <= feas_tol

    Returns y (R^m), an approximate dual vector.
    """
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape

    # Precompute thin column-space basis for fast projection
    Q = _thin_colspace_basis(A)
    def proj_null(yv: np.ndarray) -> np.ndarray:
        if Q.size == 0:
            return yv
        # y - Q (Q^T y)
        return yv - Q @ (Q.T @ yv)

    y = np.zeros(m)
    if eta is None:
        eta = 0.9 / (np.linalg.norm(b) + 1e-9)

    for t in range(1, steps + 1):
        # Ascent step on linear objective b^T y
        y += eta * b
        # Project into nullspace of A^T using fast projector
        y = proj_null(y)
        # Clip to box [-1, 1]
        np.clip(y, -1.0, 1.0, out=y)
        # Early stop if sufficiently feasible
        if t % 10 == 0:
            feas_inf = np.linalg.norm(A.T @ y, ord=np.inf)
            if feas_inf <= feas_tol:
                break

    if verbose:
        feas_inf = np.linalg.norm(A.T @ y, ord=np.inf)
        print(f"[mwu_dual_solve] steps={t}, ||A^T y||_inf ≈ {feas_inf:.2e}")
    return y


def L1_find_via_mwu(
    A: np.ndarray,
    b: np.ndarray,
    rel_threshold: float = 0.99,
    steps: int = 300,
    eta: Optional[float] = None,
) -> List[int]:
    """Use MWU-inspired dual solver to rank constraints by |y_i| and pick top set.
    Returns 0-based indices of high-stress constraints.
    """
    y = mwu_dual_solve(A, b, steps=steps, eta=eta)
    if y.size == 0:
        return []
    max_abs = float(np.max(np.abs(y)))
    if max_abs <= 1e-15:
        return []
    mask = np.abs(y) >= rel_threshold * max_abs
    return [i for i, flag in enumerate(mask) if flag]

# ---------------- Hybrid: MWU candidate filter + LP confirmation ---------------- #

def l1_error(A: np.ndarray, b: np.ndarray) -> float:
    """Return min ||A x - b||_1 via single LP."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    I = np.eye(m)
    A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n
    bounds_r = [(0, None)] * (2 * m)
    bounds = bounds_x + bounds_r
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs')
    return float(res.fun) if res.success else float('inf')


def find_conflicts_iterative(A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> List[int]:
    """Single-removal conflicts: indices i where removing row i makes error < tol."""
    m, _ = A.shape
    base = l1_error(A, b)
    if base < tol:
        return []
    hits: List[int] = []
    for i in range(m):
        A_tmp = np.delete(A, i, axis=0)
        b_tmp = np.delete(b, i)
        if l1_error(A_tmp, b_tmp) < tol:
            hits.append(i)
    return hits


def find_conflicts_hybrid(
    A: np.ndarray,
    b: np.ndarray,
    rel_threshold: float = 0.99,
    tol: float = 1e-8,
    steps: int = 200,
    eta: Optional[float] = None,
) -> List[int]:
    """Use MWU to shortlist candidates, then confirm via LP with single-row deletions."""
    # Candidate selection via MWU dual magnitudes
    cand = L1_find_via_mwu(A, b, rel_threshold=rel_threshold, steps=steps, eta=eta)
    if not cand:
        return []
    # Confirm by testing only candidates (much fewer LP calls than m)
    hits: List[int] = []
    for i in cand:
        A_tmp = np.delete(A, i, axis=0)
        b_tmp = np.delete(b, i)
        if l1_error(A_tmp, b_tmp) < tol:
            hits.append(i)
    return hits

# ---------------- Benchmark Suite ---------------- #

def create_infeasible_system(m: int, n: int, bad: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Create dense system with `bad` inconsistent rows appended. Returns (A, b, bad_indices)."""
    rng = np.random.default_rng(seed)
    x_true = rng.normal(size=(n,))
    A = rng.normal(size=(m, n))
    b = A @ x_true

    # force last `bad` rows inconsistent linear combos
    num_consistent_base = max(1, n - min(bad, n - 1))
    bad_indices: List[int] = []
    for i in range(m - bad, m):
        coeffs = rng.normal(size=(num_consistent_base,))
        new_row = coeffs @ A[:num_consistent_base, :]
        b_consistent = coeffs @ b[:num_consistent_base]
        b_inconsistent = b_consistent + (rng.random() + 0.5)
        A[i, :] = new_row
        b[i] = b_inconsistent
        bad_indices.append(i)
    return A, b, bad_indices


def benchmark(
    test_cases: List[Tuple[str, np.ndarray, np.ndarray, List[int]]],
    rel_threshold: float = 0.99,
    tol: float = 1e-8,
) -> Tuple[List[dict], Dict[str, float]]:
    """Benchmark standard (LP duals), MWU, and hybrid. Returns (rows, averages)."""
    rows: List[dict] = []
    sums: Dict[str, float] = {
        'linprog_time': 0.0,
        'mwu_time': 0.0,
        'hybrid_time': 0.0,
        'linprog_f1': 0.0,
        'mwu_f1': 0.0,
        'hybrid_f1': 0.0,
        'linprog_success': 0.0,
        'mwu_success': 0.0,
        'hybrid_success': 0.0,
    }

    def f1_score(pred: List[int], truth: List[int]) -> float:
        p = set(pred)
        t = set(truth)
        if len(p) == 0 and len(t) == 0:
            return 1.0
        if len(p) == 0 or len(t) == 0:
            return 0.0
        tp = len(p & t)
        prec = tp / max(1, len(p))
        rec = tp / max(1, len(t))
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def exact(pred: List[int], truth: List[int]) -> bool:
        return set(pred) == set(truth)

    for name, A, b, bad_idx in test_cases:
        # Baseline linprog duals
        t0 = time.perf_counter()
        pred_lin = L1_find_via_linprog(A, b, rel_threshold=rel_threshold)
        t1 = time.perf_counter() - t0

        # MWU-only duals
        t0 = time.perf_counter()
        pred_mwu = L1_find_via_mwu(A, b, rel_threshold=rel_threshold)
        t2 = time.perf_counter() - t0

        # Hybrid (MWU shortlist + confirmation)
        t0 = time.perf_counter()
        pred_hyb = find_conflicts_hybrid(A, b, rel_threshold=rel_threshold, tol=tol)
        t3 = time.perf_counter() - t0

        row = {
            'case': name,
            'm': A.shape[0],
            'n': A.shape[1],
            'truth_count': len(bad_idx),
            'linprog_pred_count': len(pred_lin),
            'mwu_pred_count': len(pred_mwu),
            'hybrid_pred_count': len(pred_hyb),
            'linprog_f1': f1_score(pred_lin, bad_idx),
            'mwu_f1': f1_score(pred_mwu, bad_idx),
            'hybrid_f1': f1_score(pred_hyb, bad_idx),
            'linprog_exact': exact(pred_lin, bad_idx),
            'mwu_exact': exact(pred_mwu, bad_idx),
            'hybrid_exact': exact(pred_hyb, bad_idx),
            'linprog_time': t1,
            'mwu_time': t2,
            'hybrid_time': t3,
        }
        rows.append(row)
        sums['linprog_time'] += t1
        sums['mwu_time'] += t2
        sums['hybrid_time'] += t3
        sums['linprog_f1'] += row['linprog_f1']
        sums['mwu_f1'] += row['mwu_f1']
        sums['hybrid_f1'] += row['hybrid_f1']
        sums['linprog_success'] += 1.0 if row['linprog_exact'] else 0.0
        sums['mwu_success'] += 1.0 if row['mwu_exact'] else 0.0
        sums['hybrid_success'] += 1.0 if row['hybrid_exact'] else 0.0

    n = max(1, len(test_cases))
    avgs = {k: v / n for k, v in sums.items()}
    return rows, avgs


# ---------------- Demo / CLI ---------------- #
if __name__ == "__main__":
    # Build a few test cases
    cases: List[Tuple[str, np.ndarray, np.ndarray, List[int]]] = []
    sizes = [(10000, 100)]
    seed = 1
    for m, n in sizes:
        for bad in [1, 2, 3]:
            A, b, bad_idx = create_infeasible_system(m, n, bad, seed=seed)
            seed += 1
            cases.append((f"dense_{m}x{n}_bad{bad}", A, b, bad_idx))

    rows, avgs = benchmark(cases)

    # Simplified, clear report: times and correctness only
    print("Benchmark summary (times and exact correctness):")
    print("-" * 72)
    for r in rows:
        print(f"Case: {r['case']} (m={r['m']}, n={r['n']}, truth={r['truth_count']})")
        print(f"  linprog: time={r['linprog_time']:.6f}s, correct={r['linprog_exact']}")
        print(f"  mwu    : time={r['mwu_time']:.6f}s, correct={r['mwu_exact']}")
        print(f"  hybrid : time={r['hybrid_time']:.6f}s, correct={r['hybrid_exact']}")
        print()

    n = max(1, len(rows))
    print("Averages:")
    print(f"  linprog: time={avgs['linprog_time']:.6f}s, success_rate={avgs['linprog_success']:.2f}")
    print(f"  mwu    : time={avgs['mwu_time']:.6f}s, success_rate={avgs['mwu_success']:.2f}")
    print(f"  hybrid : time={avgs['hybrid_time']:.6f}s, success_rate={avgs['hybrid_success']:.2f}")
