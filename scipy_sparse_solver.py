import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class SciPySparseSolver:
    """
    Solves the linear system Ax=b using scipy.sparse.linalg.spsolve.

    This solver is designed for square, non-singular systems. It will
    gracefully handle non-square or singular matrices by returning a
    solution vector of NaNs and appropriate status information.
    """
    def __init__(self, A, b):
        """
        Initializes the solver with the matrix A and vector b.

        Args:
            A (array-like or sparse matrix): The coefficient matrix of the system.
            b (array-like): The right-hand side vector of the system.
        """
        # Ensure A is in a suitable sparse format (CSC is good for solvers)
        if not sp.issparse(A):
            self.A = sp.csc_matrix(A)
        else:
            self.A = A.tocsc()
        
        # Ensure b is a NumPy array
        self.b = np.asarray(b)
        self.m, self.n = self.A.shape

    def solve(self):
        """
        Attempts to solve the linear system Ax=b.

        Returns:
            tuple: A tuple containing:
                - solution (np.ndarray): The solution vector x. If the system cannot be
                  solved, this is an array filled with np.nan.
                - info (dict): A dictionary containing metadata about the solve process,
                  including a 'status' and 'message'.
        """
        info = {
            'solver': 'scipy.sparse.linalg.spsolve',
            'matrix_shape': (self.m, self.n)
        }

        # SciPy's spsolve requires a square matrix for a direct solve.
        if self.m != self.n:
            info['status'] = 'failed_non_square'
            info['message'] = f"Matrix must be square to solve, but has shape {(self.m, self.n)}."
            solution = np.full(self.n, np.nan)
            return solution, info

        try:
            # The core of the solver
            solution = spsolve(self.A, self.b)
            info['status'] = 'success'
            info['message'] = 'System solved successfully.'
        
        # spsolve may raise an exception if the matrix is singular.
        except RuntimeError as e:
            info['status'] = 'failed_singular'
            info['message'] = f"Solver failed. The matrix is likely singular. Error: {e}"
            solution = np.full(self.n, np.nan)

        return solution, info