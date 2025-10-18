import numpy as np
from solver_utils import Solver
import scipy as sp


def find_iis_farkas(A, b, tol=1e-8):
    """
    Finds an Irreducible Inconsistent Subsystem (IIS) for the system Ax = b using farkas lemma principle
    """
    if Solver.is_consistent(A, b):
        return []

    At_dense = A.T.toarray()
    left_null_space_basis = sp.linalg.null_space(At_dense)

    if left_null_space_basis.shape[1] == 0:

        iis_candidate_indices = np.arange(A.shape[0])
    else:
        b_projections = np.abs(left_null_space_basis.T @ b)
        best_basis_vector_idx = np.argmax(b_projections)
        y = left_null_space_basis[:, best_basis_vector_idx]
        
    iis_candidate_indices = np.where(np.abs(y) > tol)[0]
    essential_indices = []

    for i in iis_candidate_indices:
        subset_to_test = [j for j in iis_candidate_indices if j != i]
        
        A_sub = A[subset_to_test, :]
        b_sub = b[subset_to_test]
        if Solver.is_consistent(A_sub, b_sub):
            essential_indices.append(i)
    return sorted(essential_indices)