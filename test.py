from scipy.sparse.linalg import spsolve
import time 
import numpy as np
import scipy.sparse as sp
def make_spd_banded(n: int):
    # 5-diagonal, strictly diagonally dominant SPD matrix
    main = 4.0 * np.ones(n)
    off1 = -1.0 * np.ones(n - 1)
    off2 = -0.25 * np.ones(n - 2)
    return sp.diags([off2, off1, main, off1, off2], offsets=[-2, -1, 0, 1, 2], format='csr')

# Benchmark spsolve (sparse) vs np.linalg.solve (dense)
sizes = [500, 1000]  # increase carefully (dense O(n^3) can be slow)
repeats = 3
rng = np.random.default_rng(123)

for n in sizes:
    A_mat = make_spd_banded(n)
    x_true = rng.standard_normal(n)
    b_vec = A_mat @ x_true

    # Sparse solve
    t_best_sparse = float('inf')
    x_sparse = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        x_candidate = spsolve(A_mat, b_vec)
        t = time.perf_counter() - t0
        if t < t_best_sparse:
            t_best_sparse = t
            x_sparse = x_candidate

    # Dense solve (convert once)
    t0 = time.perf_counter()
    A_dense = A_mat.toarray()
    t_dense_convert = time.perf_counter() - t0

    t_best_dense = float('inf')
    x_dense = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        x_candidate = np.linalg.solve(A_dense, b_vec)
        t = time.perf_counter() - t0
        if t < t_best_dense:
            t_best_dense = t
            x_dense = x_candidate

    # Accuracy checks
    rel_res_sparse = np.linalg.norm(A_mat @ x_sparse - b_vec) / np.linalg.norm(b_vec)
    rel_res_dense = np.linalg.norm(A_dense @ x_dense - b_vec) / np.linalg.norm(b_vec)

    nnz = A_mat.nnz
    density = nnz / (n * n)
    print(f"n={n} | nnz={nnz} ({density:.4%} dense)")
    print(f"  spsolve:      {t_best_sparse:.4f}s | rel_res={rel_res_sparse:.2e}")
    print(f"  np.linalg:    {t_best_dense:.4f}s (+{t_dense_convert:.4f}s to densify) | rel_res={rel_res_dense:.2e}")
    if t_best_sparse > 0:
        print(f"  speedup (dense/sparse) ~ {t_best_dense / t_best_sparse:.2f}x")
    print()