import numpy as np
import scipy.sparse as sp
import sparseqr
from scipy.linalg import solve_triangular

class QR_find_class:
    """
    Finds inconsistent linear constraints using sparse QR factorization.

    This class implements the algorithm to identify which constraints (rows) in a
    linear system Ax = b are inconsistent with a "core" set of linearly
    independent constraints.

    Attributes:
        A (scipy.sparse.csc_matrix): The constraint matrix.
        b (np.ndarray): The right-hand side vector.
        m (int): The number of constraints (rows in A).
        n (int): The number of variables (columns in A).
        conflicts (dict): The results of the analysis, populated after solve() is called.
    """
    def __init__(self, A, b):
        """
        Initializes the QR_find_class with a sparse matrix A and vector b.

        Args:
            A (array-like): The m x n constraint matrix. Can be dense or sparse.
            b (array-like): The m x 1 right-hand side vector.
        """
        # Ensure A is in a sparse format suitable for column operations (CSC)
        self.A = sp.csc_matrix(A)
        self.b = np.asarray(b).flatten()
        self.m, self.n = self.A.shape

        if self.m != len(self.b):
            raise ValueError("Shape mismatch: A has {} rows but b has {} elements.".format(self.m, len(self.b)))

        self.conflicts = None

    def solve(self, tol=1e-8):
        """
        Analyzes the system Ax = b to find inconsistent constraints.

        The method performs a rank-revealing QR factorization on A.T to identify
        a consistent core set of constraints and then checks remaining (dependent)
        constraints against this core. An orthogonality filter is applied to
        report only geometrically meaningful conflicts.

        Args:
            tol (float): The numerical tolerance for determining rank, zero values,
                         and orthogonality.

        Returns:
            dict: A dictionary where keys are the indices of inconsistent ("bad")
                  constraints and values are a list of the indices of the core
                  constraints they conflict with.
        """
        # If there are no constraints, there are no conflicts.
        asf = np.dot(A.T, A)
        if self.m == 0:
            return {}

        # Step 1: Analyze constraints by performing QR on the transpose of A.
        # The columns of A.T are the constraints (rows of A).
        # sparseqr.qr returns Q, R, E (permutation vector), and the numerical rank.
        # 'E' contains the new order of columns from A.T (i.e., new order of constraints).
        Q, R, E, rank = sparseqr.qr(self.A.T)
        print("Done with QR")
        # If all constraints are linearly independent, the system cannot be over-determined
        # in a way that creates a conflict among the rows. It is consistent.
        if rank == self.m:
            self.conflicts = {}
            return self.conflicts

        # Step 2: Identify the "Consistent Core" and Dependent sets using the rank.
        # The first `rank` constraints in the permuted order form the core.
        b_perm = self.b[E]  # Reorder b according to the permutation E

        # Partition the reordered b and the R matrix
        b_B = b_perm[:rank]  # Right-hand side for the "Basic" or "core" set
        b_D = b_perm[rank:] # Right-hand side for the "Dependent" set

        R11 = R.tocsr()[:rank, :rank]
        R12 = R.tocsr()[:rank, rank:]

        # Step 3: Determine Dependencies via the matrix L.
        # The relationship is A_D.T = A_B.T * L.T, where L.T = inv(R11) * R12.
        # We solve the triangular system R11 * X = R12 for X instead of inverting.
        # Note: solve_triangular requires dense arrays.
        X = solve_triangular(R11.toarray(), R12.toarray(), lower=False)
        print("Done w Triangular")
        L = X.T

        # Step 4: Check for Consistency by predicting b_D.
        # For the system to be consistent, we must have b_D = L @ b_B.
        predicted_b_D = L @ b_B
        residuals = b_D - predicted_b_D

        # Step 5: Flag Inconsistencies based on residuals.
        # Find the indices of dependent constraints where the residual is significant.
        bad_dependent_indices = np.where(np.abs(residuals) > tol)[0]

        # Step 6: Pinpoint Specific Conflicts with the Orthogonality Filter.
        conflicts_dict = {}
        for i in bad_dependent_indices:
            # Get the original index of the bad constraint
            original_bad_idx = E[rank + i]
            
            # Find which core constraints it depends on from the L matrix
            core_dependencies_perm_indices = np.where(np.abs(L[i, :]) > tol)[0]

            conflicting_core_indices = []
            bad_vec = self.A[original_bad_idx, :]

            for j in core_dependencies_perm_indices:
                # Get the original index of the core constraint
                original_core_idx = E[j]
                core_vec = self.A[original_core_idx, :]

                # Orthogonality Check:
                # Calculate dot product. If they are orthogonal, their conflict is not
                # geometrically direct (e.g., x=5 vs y=10). We ignore these.
                dot_product = np.abs(bad_vec.dot(core_vec.T).toarray()[0, 0])

                if dot_product > tol:
                    conflicting_core_indices.append(original_core_idx)
            
            if conflicting_core_indices:
                # Only add to dict if there are non-orthogonal conflicts
                conflicts_dict[original_bad_idx] = sorted(conflicting_core_indices)

        self.conflicts = conflicts_dict
        return self.conflicts


# Our four constraints:
# 0: x = 0
# 1: y = 0
# 2: x + y = 10
# 3: x + y = 12

A = sp.csr_matrix([
    [1, 0, 0],
    [1, 1, 1],
    [2, 1, 1],
    [1, 0, 1]
])

b = np.array([10, 5, 15, 12])

rng = np.random.default_rng(42)
A = sp.random(1100, 1000, density=0.05).tocsr()
b = rng.random(1100)

A = sp.csr_matrix([
    [3, 3, 0],
    [3, -3, 0],
    [1, 0, 0],
    [1, 0, 0],
    [6, 6, 0],
    [0, 0, 1]
])

b = np.array([3, 1.5, 1, -1, 6, 1])

# Instantiate the class and solve
qr_finder = QR_find_class(A, b)
inconsistent_sets = qr_finder.solve()

# --- Print the results in a readable way ---
print("Analysis of Inconsistent Constraints\n" + "="*40)

if not inconsistent_sets:
    print("The system is consistent.")
else:
    for bad_idx, conflict_indices in inconsistent_sets.items():
        print(f"Constraint {bad_idx} is INCONSISTENT.")
        print(f"   It conflicts with the core constraints at indices: {conflict_indices}\n")

# Let's verify the logic
# The algorithm should identify that constraint 2 (x+y=10) and constraint 3 (x+y=12)
# are both inconsistent with the core basis established by constraints 0 (x=0) and 1 (y=0).
# The expected output should flag 2 and 3 and link them to 0 and 1.