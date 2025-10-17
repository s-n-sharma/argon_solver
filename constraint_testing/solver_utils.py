import numpy as np
import scipy.sparse as sp
import sparseqr
from numpy.linalg import norm
from scipy.linalg import solve_triangular
class Solver:
    """When running solvers use the following convention:
    -2 under constrained and conflicting 
    -1 under constrained and consistent
    0 
    """
    
    def create_augmenting_system(A,b):
        """creates A' for A'x = 0"""
        
        b_col = -b.reshape(-1, 1)
        b_sparse = sp.csc_matrix(b_col)
        A_prime = sp.hstack([A, b_sparse], format='csc')
        
        return A_prime
    def graph_laplacian(A_prime):
        
        S = A_prime.copy()
        S.data[:] = 1  
        S = S.tocsc()

        row_degrees = S.sum(axis=1).A1  
        Dc = sp.diags(row_degrees, format='csc')

        col_degrees = S.sum(axis=0).A1
        Dv = sp.diags(col_degrees, format='csc')

        L = sp.bmat([
            [Dc, -S],
            [-S.transpose(), Dv]
        ], format='csc')
        
        return L


    def analyze_system_with_qr(A, b):
        """
        Analyzes the system Ax=b using sparseqr.qr function.
        """
        m, n = A.shape
        if not sp.isspmatrix_coo(A):
            A = A.tocoo()

        try:
            Q, R, E, rank = sparseqr.qr(
                A, economy=False, tolerance=1e-10
            )
        except Exception as e:
            print(f"QR solver failed: {e}")
            return
        b_col = b.reshape(-1, 1)
        c_vec = Q.transpose() @ b_col
        c_bottom = c_vec[rank:]
        
        conflict_norm = norm(c_bottom)


        is_unconstrained = rank < n
        is_conflicting = conflict_norm > 1e-9 
        

        c = Q.T @ b
        
        # get solvable parts 
        c1 = c[:rank]
        R11 = R.tocsc()[:rank, :rank] 

    
        if sp.issparse(R11):
            R11 = R11.toarray() 
            
        y1 = solve_triangular(R11, c1, lower=False)

        x_hat = np.zeros(n)
        x_hat[E[:rank]] = y1
        residual = b - A @ x_hat

        sorted_indices = np.argsort(np.abs(residual))[::-1]
        
        return sorted_indices, residual

    def is_conflicting_using_QR(A,b):
        """wrapper to do test conflicting constraints"""
       
        if not sp.isspmatrix_coo(A):
            A = A.tocoo()
        b = b.reshape(-1, 1)
            
        try:
            _= sparseqr.solve(
                A, b
            )
            return _
        except Exception as e:
            print(f"QR solver failed: {e}")
            return