import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import time


def _givens(a, b):
    if b == 0.0:
        return 1.0, 0.0, a
    if abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1 + tau * tau)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1 + tau * tau)
        s = c * tau
    r = c * a - s * b
    return c, s, r


essential_eps = 1e-10


def _apply_givens_to_rows(R, i, k, c, s, start_col):
    ri = R[i, start_col:]
    rk = R[k, start_col:]
    R[i, start_col:] = c * ri - s * rk
    R[k, start_col:] = s * ri + c * rk


def _apply_givens_to_Q_columns(Q, i, k, c, s):
    qi = Q[:, i].copy()
    qk = Q[:, k].copy()
    Q[:, i] = c * qi - s * qk
    Q[:, k] = s * qi + c * qk


def qr_with_col_pivot(A_dense):
    Q, R, P = la.qr(A_dense, mode="full", pivoting=True)
    return Q, R, P


def update_add_row(R, Q, P, row_original):
    n = R.shape[1]
    row_perm = row_original[:, P]
    R = np.vstack([R, row_perm])
    Q = np.pad(Q, ((0, 1), (0, 1)))
    Q[-1, -1] = 1.0
    last = R.shape[0] - 1
    upto = min(n - 1, last - 1)
    for j in range(0, upto + 1):
        a = R[j, j]
        b = R[last, j]
        if abs(b) < essential_eps:
            continue
        c, s, _ = _givens(a, b)
        _apply_givens_to_rows(R, j, last, c, s, j)
        _apply_givens_to_Q_columns(Q, j, last, c, -s)
    return R, Q


def update_add_col(R, Q, col_original):
    y = Q.T @ col_original
    R = np.hstack([R, y.reshape(-1, 1)])
    m, n_new = R.shape
    new_col = n_new - 1
    last_row_used = min(m - 1, new_col)
    for i in range(last_row_used, 0, -1):
        a = R[i - 1, new_col]
        b = R[i, new_col]
        if abs(b) < essential_eps:
            continue
        c, s, _ = _givens(a, b)
        _apply_givens_to_rows(R, i - 1, i, c, s, i - 1)
        _apply_givens_to_Q_columns(Q, i - 1, i, c, -s)
    return R, Q


def build_sparse_matrix(m, n, density=0.01, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=density, format="csr", random_state=rng)
    if m == n:
        diag_idx = np.arange(n)
        A = A + sp.csr_matrix((np.ones(n) * 1e-3, (diag_idx, diag_idx)), shape=(m, n))
    return A


def benchmark_updates(m0=800, n0=800, density=0.01, batch_sizes=(1, 2, 4, 8, 16, 32), redundant_every=4, seed=0):
    rng = np.random.default_rng(seed)
    try:
        import sparseqr as spqr  # noqa: F401
        use_sparse_qr = True
    except Exception:
        use_sparse_qr = False
    if not use_sparse_qr and (m0 > 500 or n0 > 500):
        m0 = min(m0, 400)
        n0 = min(n0, 400)
    A = build_sparse_matrix(m0, n0, density=density, seed=seed)
    A_dense = A.toarray()
    t0 = time.perf_counter()
    Q, R, P = qr_with_col_pivot(A_dense)
    t_qr = time.perf_counter() - t0
    print("Initial QR (full, with column pivoting)")
    print(f"m={m0}, n={n0}, density={density}, time={t_qr:.3f}s")
    A_perm = A_dense[:, P]
    err0 = la.norm(Q @ R - A_perm) / max(1.0, la.norm(A_perm))
    print(f"Initial relative reconstruction error: {err0:.2e}")
    current_m, current_n = m0, n0
    for b_idx, bsz in enumerate(batch_sizes, start=1):
        new_rows = sp.random(bsz, current_n, density=density, format="csr", random_state=rng).toarray()
        if b_idx % redundant_every == 0 and current_m >= 2:
            alpha = rng.standard_normal()
            beta = rng.standard_normal()
            combo = alpha * A_dense[rng.integers(0, current_m)] + beta * A_dense[rng.integers(0, current_m)]
            new_rows[0, :] = combo
        new_cols_block = sp.random(current_m + bsz, bsz, density=density, format="csr", random_state=rng).toarray()
        if b_idx % redundant_every == 0 and current_n >= 2:
            gamma = rng.standard_normal()
            delta = rng.standard_normal()
            col_combo = gamma * A_dense[:, rng.integers(0, current_n)] + delta * A_dense[:, rng.integers(0, current_n)]
            new_cols_block[: current_m, 0] = col_combo
        t1 = time.perf_counter()
        for i in range(bsz):
            R, Q = update_add_row(R, Q, P, new_rows[i : i + 1, :])
        t_rows = time.perf_counter() - t1
        A_dense = np.vstack([A_dense, new_rows])
        current_m += bsz
        A_perm = A_dense[:, P]
        err_rows = la.norm(Q @ R - A_perm) / max(1.0, la.norm(A_perm))
        print(f"After adding {bsz} row(s): dt={t_rows:.3f}s, rel.err={err_rows:.2e}")
        new_orig_indices = np.arange(A_dense.shape[1], A_dense.shape[1] + bsz)
        t2 = time.perf_counter()
        for j in range(bsz):
            col = np.zeros((current_m,))
            col[: current_m - bsz] = new_cols_block[: current_m - bsz, j]
            col[current_m - bsz : current_m] = new_cols_block[current_m - bsz : current_m, j]
            R, Q = update_add_col(R, Q, col)
        t_cols = time.perf_counter() - t2
        add_cols = np.zeros((current_m, bsz))
        add_cols[: current_m - bsz, :] = new_cols_block[: current_m - bsz, :]
        add_cols[current_m - bsz :, :] = new_cols_block[current_m - bsz :, :]
        A_dense = np.hstack([A_dense, add_cols])
        P = np.hstack([P, new_orig_indices])
        current_n += bsz
        A_perm = A_dense[:, P]
        err_cols = la.norm(Q @ R - A_perm) / max(1.0, la.norm(A_perm))
        print(f"After adding {bsz} col(s): dt={t_cols:.3f}s, rel.err={err_cols:.2e}")
    return


if __name__ == "__main__":
    benchmark_updates()

