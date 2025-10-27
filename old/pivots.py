import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# ===================================================================
# 1. SETUP: Define matrix properties and generate data
# ===================================================================
m = 1000  # Number of constraints (rows)
n = 1200  # Number of variables (columns)
density = 0.01 # Sparsity (1% of elements are non-zero)

print("--- System Parameters ---")
print(f"Matrix Dimensions: {m} rows, {n} columns")
print(f"Sparsity: {density*100:.1f}%")
print("-" * 25)

# Generate a single sparse matrix for all tests
print("Generating sparse matrix...")
A_sparse = sp.random(m, n, density=density, format='csc', dtype=np.float64)
print("Sparse matrix created.")

# Create a dense version for the NumPy benchmark
print("Generating dense matrix from sparse data...")
A_dense = A_sparse.toarray()
memory_mb = A_dense.nbytes / (1024**2)
print(f"Dense matrix created. Memory usage: {memory_mb:.2f} MB")
print("=" * 25 + "\n")


# ===================================================================
# 2. BENCHMARK 1: Sparse SVD (scipy.svds)
# ===================================================================
print("--- 🔬 Method 1: Sparse SVD (`scipy.sparse.linalg.svds`) ---")
start_time = time.perf_counter()

# k must be less than min(shape). We get almost all to find the rank.
k = min(A_sparse.shape) - 1
_u, s, _vt = svds(A_sparse, k=k)

# Rank is the number of singular values greater than a small tolerance
tolerance = 1e-10
rank_svd = np.sum(s > tolerance)

duration_ms = (time.perf_counter() - start_time) * 1000
num_free_vars_svd = n - rank_svd

print(f"Execution Time: {duration_ms:.2f} ms")
print(f"Matrix Rank Found: {rank_svd}")
print(f"Number of Free Variables: {num_free_vars_svd}")
print("(Note: This method is less direct for identifying *which* variables are free)")
print("-" * 25 + "\n")


# ===================================================================
# 3. BENCHMARK 2: Dense QR (numpy.linalg.qr)
# ===================================================================
print("--- 🐢 Method 2: Dense QR (`numpy.linalg.qr`) ---")
start_time = time.perf_counter()

# Perform the dense QR decomposition
_Q_dense, R_dense = np.linalg.qr(A_dense)

# Rank is the number of non-negligible diagonal elements in R
rank_dense = np.sum(np.abs(R_dense.diagonal()) > tolerance)

duration_ms = (time.perf_counter() - start_time) * 1000
num_free_vars_dense = n - rank_dense

print(f"Execution Time: {duration_ms:.2f} ms")
print(f"Matrix Rank Found: {rank_dense}")
print(f"Number of Free Variables: {num_free_vars_dense}")
print("-" * 25 + "\n")