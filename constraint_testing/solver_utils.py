import numpy as np

import scipy.sparse.linalg as spla
import scipy as sp
import scipy.sparse as spa 
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
        b_sparse = spa.csc_matrix(b_col)
        A_prime = spa.hstack([A, b_sparse], format='csc')
        
        return A_prime
    def graph_laplacian(A_prime):
        
        S = A_prime.copy()
        S.data[:] = 1  
        S = S.tocsc()

        row_degrees = S.sum(axis=1).A1  
        Dc = spa.diags(row_degrees, format='csc')

        col_degrees = S.sum(axis=0).A1
        Dv = spa.diags(col_degrees, format='csc')

        L = spa.bmat([
            [Dc, -S],
            [-S.transpose(), Dv]
        ], format='csc')
        
        return L


    def analyze_system_with_qr(A, b, verbose = False):
        """
        Analyzes the system Ax=b using sparseqr.qr function.
        """
        m, n = A.shape
        if not spa.isspmatrix_coo(A):
            A = spa.coo_matrix(A)

        try:
            Q, R, E, rank = sparseqr.qr(
                A, economy=False, tolerance=1e-10
            )
        except Exception as e:
            print(f"QR solver failed: {e}")
            return
        if (verbose):         
            print("------------------")
            print("R diagonal")
            print(R.toarray())
            print("------------------")
        b_col = b.reshape(-1, 1)
        c_vec = Q.transpose() @ b_col
        c_bottom = c_vec[rank:]
        
        conflict_norm = norm(c_bottom)


        is_unconstrained = rank < n
        is_conflicting = conflict_norm > 1e-9 
        

        c = Q.T @ b
        if verbose:
            print("c matrix")
            print(f"{c}")
            print("------------------")
        
        
        # get solvable parts 
        c1 = c[:rank]
        R11 = R.tocsc()[:rank, :rank] 

    
        if spa.issparse(R11):
            R11 = R11.toarray() 
            
        y1 = solve_triangular(R11, c1, lower=False)
        if verbose:
            print(y1)

        x_hat = np.zeros(n)
        x_hat[E[:rank]] = y1
        residual = np.asarray(b - A @ x_hat).reshape(-1)
        if verbose:
            print(residual)
        sorted_indices = np.argsort(np.abs(residual))[::-1]
        sorted_indices = np.asarray(sorted_indices, dtype=int).reshape(-1)
        
        return sorted_indices, residual

    def is_conflicting_using_QR(A,b):
        """
        Analyzes the system Ax=b using sparseqr.qr function.
        """
        m, n = A.shape
        if not spa.isspmatrix_coo(A):
            A = spa.coo_matrix(A)

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
        return is_conflicting
        
    def is_consistent(A_sub, b_sub):
        """
            feasibility 
        """
        if A_sub.shape[0] == 0:
            return True 

        c = np.zeros(A_sub.shape[1])
        res = sp.optimize.linprog(
            c=c,
            A_eq=A_sub,
            b_eq=b_sub,
            bounds=(None, None),
            method='highs', 
            options={'presolve': False} 
        )

        return res.status != 2
    
    def verify_iis(A, b, iis_indices):
        """verify the indices provided are good"""
        A_iis = A[iis_indices, :]
        b_iis = b[iis_indices]
        
        if Solver.is_consistent(A_iis, b_iis):
            return False
        for i in iis_indices:
            subset_indices = [j for j in iis_indices if j != i]
            A_sub = A[subset_indices, :]
            b_sub = b[subset_indices]

            if not Solver.is_consistent(A_sub, b_sub):
                return False
        return True