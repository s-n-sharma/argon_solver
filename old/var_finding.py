import numpy as np 
import scipy
import time
"""

Goal is to find the variables that are not uniquely determined in Ax = b
do this by considering the basis vectors e_i
If e_i is in the null space of A, then x_i is not uniquely determined
check if e_i has any component in the null space --> check if e_i in row space of A
--> then check V V^T e_i = e_i
where V is an orthonormal matrix whose columns are a basis for the rowspace of A,
how to get?
use QR on A^T -> Q's columns then span column space of A^T, which is row space of A
obv A wouldn't be full rank, so discard columns of Q that correspond to zero rows of R 
"""

def var_find(A, b, rel_threshold=1e-5):
    """
    Uses QR decomposition of A^T to find the variables that are not fully determined
    """
    A = np.asarray(A)
    m, n = A.shape
    
    Q, R, _ = scipy.linalg.qr(A.T, mode='economic', pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    if rank == 0:
        return list(range(n))
    
    Q1 = Q[:, :rank]
    VVT = Q1 @ Q1.T
    
    undetermined_vars = []
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        proj = VVT @ e_i
        if np.linalg.norm(proj - e_i) > rel_threshold:
            undetermined_vars.append(i)
    
    return undetermined_vars

#tests
if __name__ == "__main__":
    
    A1 = np.array([[1, 2], [2, 4], [3, 6]])
    b1 = np.array([1, 2, 3])
    print("Test Case 1 - Undetermined Variables:", var_find(A1, b1))

    A2 = np.array([[1, 1, 0], [1, -1, 0], [2, 2, 0]])
    b2 = np.array([2, 0, 4])

    print("Test Case 2 - Undetermined Variables:", var_find(A2, b2))

    print("\n Dense 1000x1000 matrices:")

    for i in range(5):
        A = np.random.rand(1000, 1000)
        b = np.random.rand(1000)

        start = time.time()







