import numpy as np
import scipy.sparse as sp
import scipy.linalg

class QRSolver:
    def __init__(self, A, b):
        if sp.issparse(A):
            self.A = A.toarray()
        else:
            self.A = np.asarray(A)
        self.b = np.asarray(b)
        self.m, self.n = self.A.shape

    def solve(self):
        solution = np.full(self.n, np.nan)
        info = {
            'status': 'unprocessed',
            'rank': 0,
            'is_consistent': False,
            'pivot_variables': [],
            'free_variables': []
        }

        if self.m == 0 or self.n == 0:
            info['status'] = 'trivial'
            return solution, info

        # 1. Perform dense QR decomposition with column pivoting
        # P is the permutation vector (indices of original columns)
        Q, R, P = scipy.linalg.qr(self.A, pivoting=True)
        
        # 2. Calculate the rank by inspecting the diagonal of R
        tol = 1e-12
        if self.m > 0 and self.n > 0:
             tol = np.max(self.A.shape) * np.finfo(R.dtype).eps * abs(R[0, 0])
        
        # The rank is the number of diagonal entries of R greater than the tolerance
        rank = np.sum(np.abs(np.diag(R)) > tol)
        info['rank'] = rank
        
        # 3. Identify pivot and free variables from the permutation vector P
        pivot_indices = P[:rank]
        free_indices = P[rank:]
        info['pivot_variables'] = sorted(pivot_indices.tolist())
        info['free_variables'] = sorted(free_indices.tolist())

        # 4. Check for inconsistency (overdetermination)
        c = Q.T @ self.b
        
        residual_norm = np.linalg.norm(c[rank:])
        is_consistent = residual_norm < 1e-9
        info['is_consistent'] = is_consistent
        info['residual_from_qr'] = residual_norm
        
        # 5. Solve for the pivot variables
        if rank > 0:
            R_pivots = R[:rank, :rank]
            c_pivots = c[:rank]
            
            x_pivots = scipy.linalg.solve_triangular(R_pivots, c_pivots, lower=False)
            
            solution[pivot_indices] = x_pivots

        # 6. Set the final status based on the analysis
        if not is_consistent:
            info['status'] = 'inconsistent'
        elif free_indices.size > 0:
            info['status'] = 'underdetermined'
        else:
            info['status'] = 'determined'

        return solution, info