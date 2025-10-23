import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.linalg import svd

# A small constant for machine precision
unit_roundoff = np.finfo(float).eps

def create_sparse_sign_embedding(m, d, zeta):
    """
    Creates a sparse sign embedding matrix S of size d x m.
    Each column of S has exactly zeta non-zero entries, which are +/- 1.
    This is the sketching matrix used in FOSSILS[cite: 110, 111, 331].
    
    Args:
        m (int): Number of rows in the original matrix A.
        d (int): The embedding dimension (number of rows for S).
        zeta (int): Number of non-zero entries per column of S.

    Returns:
        scipy.sparse.csc_matrix: The sparse sketching matrix S.
    """
    rows = np.zeros(m * zeta, dtype=int)
    cols = np.zeros(m * zeta, dtype=int)
    vals = np.zeros(m * zeta)

    for j in range(m):
        # Choose zeta random row indices without replacement
        row_indices = np.random.choice(d, zeta, replace=False)
        # Assign random +/- 1 values
        random_signs = np.random.choice([-1.0, 1.0], zeta)
        
        start, end = j * zeta, (j + 1) * zeta
        rows[start:end] = row_indices
        cols[start:end] = j
        vals[start:end] = random_signs

    # The paper mentions a scaling factor, which is sqrt(d / (m * zeta))
    # for some variants, but here we follow the simpler +/- 1 structure.
    # A scaling factor is often absorbed into the analysis.
    return csc_matrix((vals / np.sqrt(zeta), (rows, cols)), shape=(d, m))


def fossils_solver(A, b, max_iter=100):
    """
    Implementation of the FOSSILS algorithm.
    Solves the least-squares problem argmin_x ||b - Ax||.
    This implementation follows the recommended Algorithm 7 from the paper.
    """
    m, n = A.shape
    
    # --- 1. Pre-processing: Column Scaling [cite: 328] ---
    col_norms = np.linalg.norm(A, axis=0)
    # Avoid division by zero for empty columns
    col_norms[col_norms == 0] = 1.0
    A_scaled = A / col_norms

    # --- 2. Sketching and Preconditioning Setup ---
    # Set parameters for the embedding [cite: 332]
    d = 12 * n 
    zeta = 8   
    
    # Create the sketching matrix S
    S = create_sparse_sign_embedding(m, d, zeta)
    
    # Sketch the matrix and compute its SVD [cite: 337]
    SA = S @ A_scaled
    U, Sigma, Vt = svd(SA, full_matrices=False)
    V = Vt.T

    # --- 3. Handle Numerical Rank Deficiency [cite: 353] ---
    cond_est = Sigma[0] / Sigma[-1]
    mu = 0.0
    A_norm_F = np.linalg.norm(A, 'fro')

    if cond_est > 1.0 / (30 * unit_roundoff):
        print(f"Warning: Estimated condition number is high ({cond_est:.2e}). Applying regularization.")
        mu = 10 * A_norm_F * unit_roundoff
        Sigma_reg = np.sqrt(Sigma**2 + mu**2)
    else:
        Sigma_reg = Sigma
        
    # --- 4. Sketch-and-Solve Initialization [cite: 248, 1124] ---
    x = V @ (np.diag(1/Sigma) @ (U.T @ (S @ b)))

    # --- 5. Iterative Refinement ---
    # Define the preconditioner P (takes the role of R^-1 in the paper) [cite: 1128]
    P = V @ np.diag(1 / Sigma_reg)
    
    # Momentum parameters for Polyak heavy ball method [cite: 334]
    # The paper uses a heuristic eta = sqrt(n/d)
    eta_sq = n / d
    beta = eta_sq
    alpha = (1 - eta_sq)**2

    # Two steps of iterative refinement [cite: 249, 1126]
    for i in range(2):
        # Calculate the residual for the current solution x
        residual = b - A_scaled @ x
        
        # This is the right-hand side for the preconditioned normal equations
        # c = R^{-T}A^T r_i [cite: 833]
        c = P.T @ (A_scaled.T @ residual)
        
        y = c.copy()
        y_old = c.copy()

        # Inner solver: Polyak heavy ball method [cite: 241, 1134]
        for j in range(max_iter):
            # The core update rule for Polyak on the preconditioned normal equations
            # y_{i+1} = y_i + alpha(c - (R^{-T}A^T A R^{-1})y_i) + beta(y_i - y_{i-1})
            z = P @ y
            grad_term = c - (P.T @ (A_scaled.T @ (A_scaled @ z)) + mu**2 * z)
            delta = alpha * grad_term + beta * (y - y_old)
            
            y, y_old = y + delta, y
            
            # Adaptive stopping criteria [cite: 345, 351]
            if i == 0: # First refinement: stop when forward stability is likely achieved
                stop_thresh = (10 * Sigma[0] * np.linalg.norm(x) + 0.4 * cond_est * np.linalg.norm(residual)) * unit_roundoff
                if np.linalg.norm(delta) <= stop_thresh:
                    break
            elif j % 5 == 0: # Second refinement: check for backward stability every 5 iterations
                # Estimate backward error using the sketched Karlson-Waldén formula [cite: 313, 320]
                x_candidate = x + P @ y
                r_candidate = b - A_scaled @ x_candidate
                theta = A_norm_F / np.linalg.norm(b)
                
                gamma_sq = (theta**2 * np.linalg.norm(r_candidate)**2) / (1 + theta**2 * np.linalg.norm(x_candidate)**2)
                
                be_num_vec = np.linalg.inv(np.diag(Sigma**2 + gamma_sq)) @ (Vt @ A_scaled.T @ r_candidate)
                be_sk = (theta / np.sqrt(1 + theta**2 * np.linalg.norm(x_candidate)**2)) * np.linalg.norm(be_num_vec)

                if be_sk < A_norm_F * unit_roundoff:
                    break

        # Update the solution with the computed correction [cite: 1171]
        x = x + P @ y
        
    # --- 6. Finalization: Undo Column Scaling [cite: 1172] ---
    x_final = x / col_norms
    
    return x_final


### Benchmarking Framework

#Here's how you can benchmark the FOSSILS solver against a standard SciPy solver for sparse matrices.

def create_test_problem(m, n, cond_num=1e6, is_sparse=False, sparsity=0.01):
    """
    Generates a test least-squares problem with a known solution.
    """
    print(f"Creating a test problem of size {m}x{n} with condition number ~{cond_num:.1e}")
    if is_sparse:
        # Create a sparse matrix
        A = csc_matrix(np.random.randn(m, n)) * csc_matrix(np.random.rand(m, n) < sparsity)
    else:
        # Create a dense matrix
        A = np.random.randn(m, n)

    # Control the condition number
    U, _, Vt = svd(A, full_matrices=False)
    s = np.linspace(cond_num, 1, n)
    A = (U * s) @ Vt

    # Create a known solution and the right-hand side `b`
    x_true = np.random.randn(n)
    b = A @ x_true + np.random.randn(m) * 1e-6 # Add a small amount of noise
    
    return A, x_true, b

def get_circular_constraints_example(size=200):
    """Generates the 'circular constraints' example matrix and vector."""
    A = []
    a_row = [0]*size
    a_row[0], a_row[1], a_row[2] = 1, -2, 1
    for i in range(size - 2):
        A.append(np.array(a_row))
        a_row = [0] + a_row[:-1]
    
    A.append(np.array([1] + [0]*(size - 2) + [-1]))
    A.append(np.array([1] + [0]*(size - 1)))

    b = [0] * (size - 2) + [size - 1] + [0]
    b = np.array(b)
    A = np.array(A)
    x = np.array([1]*size)
    return A, x, b


if __name__ == "__main__":
    # --- Benchmark Parameters ---
    m, n = 40000, 100         # Matrix dimensions (m >> n)
    cond_num = 1e8            # Condition number of the matrix
    
    # --- Generate Problem ---
    # For sparse problems, SciPy's lsqr is a good baseline.
    # For dense problems, np.linalg.lstsq would be the baseline.
    A, x_true, b = create_test_problem(m, n, cond_num=cond_num, is_sparse=False)

    print("\n--- Benchmarking FOSSILS ---")
    start_time = time.time()
    x_fossils = fossils_solver(A, b)
    end_time = time.time()
    fossils_time = end_time - start_time
    fossils_error = np.linalg.norm(x_fossils - x_true) / np.linalg.norm(x_true)
    print(f"FOSSILS Time: {fossils_time:.4f} seconds")
    print(f"FOSSILS Relative Error: {fossils_error:.4e}")

    print("\n--- Benchmarking Baseline Solver (np.linalg.lstsq) ---")
    start_time = time.time()
    # Using NumPy's built-in direct solver, which often uses QR decomposition
    x_baseline, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    end_time = time.time()
    baseline_time = end_time - start_time
    baseline_error = np.linalg.norm(x_baseline - x_true) / np.linalg.norm(x_true)
    print(f"Baseline Time: {baseline_time:.4f} seconds")
    print(f"Baseline Relative Error: {baseline_error:.4e}")
    
    # --- Print Comparison ---
    print("\n--- Comparison ---")
    if fossils_time < baseline_time:
        speedup = baseline_time / fossils_time
        print(f"FOSSILS was {speedup:.2f}x faster than the baseline solver.")
    else:
        slowdown = fossils_time / baseline_time
        print(f"FOSSILS was {slowdown:.2f}x slower than the baseline solver.")

    
    A, x_true, b = get_circular_constraints_example(size=2000)

    print("\n--- Benchmarking FOSSILS ---")
    start_time = time.time()
    x_fossils = fossils_solver(A, b)
    end_time = time.time()
    fossils_time = end_time - start_time
    fossils_error = np.linalg.norm(x_fossils - x_true) / np.linalg.norm(x_true)
    print(f"FOSSILS Time: {fossils_time:.4f} seconds")
    print(f"FOSSILS Relative Error: {fossils_error:.4e}")

    print("\n--- Benchmarking Baseline Solver (np.linalg.lstsq) ---")
    start_time = time.time()
    # Using NumPy's built-in direct solver, which often uses QR decomposition
    x_baseline, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    end_time = time.time()
    baseline_time = end_time - start_time
    baseline_error = np.linalg.norm(x_baseline - x_true) / np.linalg.norm(x_true)
    print(f"Baseline Time: {baseline_time:.4f} seconds")
    print(f"Baseline Relative Error: {baseline_error:.4e}")
    
    # --- Print Comparison ---
    print("\n--- Comparison ---")
    if fossils_time < baseline_time:
        speedup = baseline_time / fossils_time
        print(f"FOSSILS was {speedup:.2f}x faster than the baseline solver.")
    else:
        slowdown = fossils_time / baseline_time
        print(f"FOSSILS was {slowdown:.2f}x slower than the baseline solver.")