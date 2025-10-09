import numpy as np
import scipy
from scipy import sparse
import scipy.sparse as sp
import time

n = 5000

def checker(a, b):
    return np.allclose(a, b)

np.random.seed(42)

A = sparse.random(n, n, density=0.01, format="csc",data_rvs=np.random.randn)
A = (A + A.T) / 2
x = np.eye(n)


np_start = time.time()
np_x = np.linalg.solve(A.toarray().astype(float), x)
np_end = time.time()
np_time = np_end - np_start
x = sp.eye(n, format="csc")

scipy_start = time.time()
scipy_x = sp.linalg.spsolve(A, x)
scipy_end = time.time()
scipy_time = scipy_end - scipy_start   


print(f"NumPy time: {np_time:.6f} seconds")
print(f"SciPy time: {scipy_time:.6f} seconds")

