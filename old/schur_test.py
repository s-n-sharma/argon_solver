import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import concurrent.futures

# --- 1. Problem Setup (Same as before) ---

def generate_poisson_problem(grid_size):
    """
    Generates the sparse matrix A and vector b for a 2D Poisson problem.
    """
    N = grid_size * grid_size
    D = sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size, grid_size))
    A = sp.kron(sp.identity(grid_size), D) + sp.kron(D, sp.identity(grid_size))
    A = A.tocsr()
    b = np.random.rand(N)
    return A, b

# --- 2. The Parallel Nested Dissection Algorithm ---

RECURSION_BASE_CASE_SIZE = 100 # Below this size, use a direct solver

def nested_dissection_solve(A, b, node_map=None, executor=None):
    """
    Solves Ax=b using a recursive nested dissection algorithm.
    Can run in parallel if a ProcessPoolExecutor is provided.
    """
    # For this implementation, we'll work with dense numpy arrays in the recursion
    if sp.issparse(A):
        A = A.toarray()

    n = A.shape[0]

    # --- Base Case ---
    if n <= RECURSION_BASE_CASE_SIZE:
        if n > 0:
            return np.linalg.solve(A, b)
        else:
            return np.array([])

    # --- 1. Partition Step (Geometric) ---
    if node_map is None:
        grid_dim = int(np.sqrt(n))
        coords = np.array(np.meshgrid(np.arange(grid_dim), np.arange(grid_dim))).T.reshape(-1, 2)
        node_map = {i: tuple(coords[i]) for i in range(n)}

    coords = np.array(list(node_map.values()))
    min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
    dims = max_coords - min_coords
    split_axis = 0 if dims[0] >= dims[1] else 1
    split_val = min_coords[split_axis] + dims[split_axis] // 2

    original_indices = list(node_map.keys())
    s_indices = [i for i, c in node_map.items() if c[split_axis] == split_val]
    b1_indices = [i for i, c in node_map.items() if c[split_axis] < split_val]
    b2_indices = [i for i, c in node_map.items() if c[split_axis] > split_val]

    # --- 2. Permutation Step ---
    perm = np.array(b1_indices + b2_indices + s_indices, dtype=int)
    inv_perm = np.argsort(perm)

    A_perm = A[perm, :][:, perm]
    b_perm = b[perm]

    n1, n2, ns = len(b1_indices), len(b2_indices), len(s_indices)
    
    # Extract blocks as dense numpy arrays
    A11, A22, Ass = A_perm[:n1, :n1], A_perm[n1:n1+n2, n1:n1+n2], A_perm[n1+n2:, n1+n2:]
    A1s, A2s = A_perm[:n1, n1+n2:], A_perm[n1:n1+n2, n1+n2:]
    As1, As2 = A_perm[n1+n2:, :n1], A_perm[n1+n2:, n1:n1+n2]
    b1, b2, bs = b_perm[:n1], b_perm[n1:n1+n2], b_perm[n1+n2:]

    # Create new node maps for subproblems
    map1 = {i: node_map[original_idx] for i, original_idx in enumerate(b1_indices)}
    map2 = {i: node_map[original_idx] for i, original_idx in enumerate(b2_indices)}

    # --- 3. Recursive Elimination (Parallel part) ---
    if executor:
        # Submit independent tasks to the process pool
        future1 = executor.submit(nested_dissection_solve, A11, b1, map1, executor)
        future2 = executor.submit(nested_dissection_solve, A22, b2, map2, executor)

        # To solve A11*X = A1s, we solve for each column in parallel
        cols_A1s = [A1s[:, j] for j in range(ns)]
        cols_A2s = [A2s[:, j] for j in range(ns)]

        # executor.map is perfect for applying a function to a list of items
        A11_inv_A1s_cols = list(executor.map(lambda c: nested_dissection_solve(A11, c, map1, executor), cols_A1s))
        A22_inv_A2s_cols = list(executor.map(lambda c: nested_dissection_solve(A22, c, map2, executor), cols_A2s))
        
        # Collect results
        A11_inv_b1 = future1.result()
        A22_inv_b2 = future2.result()
        A11_inv_A1s = np.vstack(A11_inv_A1s_cols).T if ns > 0 else np.zeros((n1, 0))
        A22_inv_A2s = np.vstack(A22_inv_A2s_cols).T if ns > 0 else np.zeros((n2, 0))
    else: # Sequential execution
        A11_inv_b1 = nested_dissection_solve(A11, b1, map1)
        A22_inv_b2 = nested_dissection_solve(A22, b2, map2)
        A11_inv_A1s_cols = [nested_dissection_solve(A11, A1s[:, j], map1) for j in range(ns)]
        A22_inv_A2s_cols = [nested_dissection_solve(A22, A2s[:, j], map2) for j in range(ns)]
        A11_inv_A1s = np.vstack(A11_inv_A1s_cols).T if ns > 0 else np.zeros((n1, 0))
        A22_inv_A2s = np.vstack(A22_inv_A2s_cols).T if ns > 0 else np.zeros((n2, 0))

    # Form the Schur complement system
    S = Ass - (As1 @ A11_inv_A1s) - (As2 @ A22_inv_A2s)
    bs_tilde = bs - As1 @ A11_inv_b1 - As2 @ A22_inv_b2

    # --- 4. Solve for Separator ---
    xs = np.linalg.solve(S, bs_tilde) if ns > 0 else np.array([])
    
    # --- 5. Back-Substitution (Parallel part) ---
    b1_prime = b1 - A1s @ xs
    b2_prime = b2 - A2s @ xs
    
    if executor:
        future1 = executor.submit(nested_dissection_solve, A11, b1_prime, map1, executor)
        future2 = executor.submit(nested_dissection_solve, A22, b2_prime, map2, executor)
        x1 = future1.result()
        x2 = future2.result()
    else: # Sequential execution
        x1 = nested_dissection_solve(A11, b1_prime, map1)
        x2 = nested_dissection_solve(A22, b2_prime, map2)

    # --- 6. Combine and Un-permute ---
    x_perm = np.concatenate([x1, x2, xs])
    x = x_perm[inv_perm]
    return x

# --- 3. Benchmarking ---

def run_benchmark(grid_size):
    print(f"\n--- Running Benchmark for a {grid_size}x{grid_size} Grid ({grid_size**2} variables) ---")
    A_sparse, b = generate_poisson_problem(grid_size)

    # --- SciPy Sparse Direct Solver ---
    print("1. SciPy Sparse Solver (spsolve)...")
    start_time = time.time()
    x_scipy = spsolve(A_sparse, b)
    print(f"   Time: {time.time() - start_time:.4f} seconds")

    # --- Sequential Nested Dissection ---
    print("2. Nested Dissection (Sequential)...")
    start_time = time.time()
    x_nd_seq = nested_dissection_solve(A_sparse.copy(), b.copy())
    print(f"   Time: {time.time() - start_time:.4f} seconds")

    # --- Parallel Nested Dissection ---
    print("3. Nested Dissection (Parallel)...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        x_nd_par = nested_dissection_solve(A_sparse.copy(), b.copy(), executor=executor)
    print(f"   Time: {time.time() - start_time:.4f} seconds")

    # --- Verification ---
    print("\nVerifying solutions...")
    print(f"   ||x_scipy - x_nd_seq||: {np.linalg.norm(x_scipy - x_nd_seq):.2e}")
    print(f"   ||x_scipy - x_nd_par||: {np.linalg.norm(x_scipy - x_nd_par):.2e}")
    assert np.allclose(x_scipy, x_nd_par, atol=1e-6)
    print("   ✅ Solutions are consistent.")

if __name__ == '__main__':
    # For smaller sizes, parallel overhead dominates.
    # A speedup is more likely for sizes > 50, but it will be slow to run.
    GRID_DIM = 40
    run_benchmark(GRID_DIM)