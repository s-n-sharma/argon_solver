import numpy as np
import scipy.sparse
import random
import time


class SilentConstraintManager:
    """
    A non-verbose version of the ConstraintManager that processes constraints
    and reports conflicts without printing step-by-step progress.
    """

    def __init__(self, n_vars):
        self.n_vars = n_vars
        self._constraints_count = 0
        self.Q = np.eye(0)
        self.R_aug = np.empty((0, n_vars + 1))
        self.TOL = 1e-8

    def add_constraint(self, a_row, b_val):
        """
        Attempts to add a new constraint. If inconsistent, it returns the
        conflicting set of indices; otherwise, it returns an empty list.
        """
        a_row = np.array(a_row, dtype=float)
        m = self._constraints_count
        n = self.n_vars

        # Perform a "dry run" of the Givens update on temporary matrices
        temp_Q = np.eye(m + 1)
        temp_Q[:m, :m] = self.Q

        new_row = np.hstack([a_row, [b_val]])
        temp_R_aug = np.vstack([self.R_aug, new_row])

        # Apply Givens rotations
        for j in range(min(m, n)):
            a, b = temp_R_aug[j, j], temp_R_aug[m, j]
            if abs(b) < self.TOL:
                continue

            r = np.hypot(a, b)
            c, s = a / r, b / r

            temp_R_aug[j, :], temp_R_aug[m, :] = (
                c * temp_R_aug[j, :] + s * temp_R_aug[m, :],
                -s * temp_R_aug[j, :] + c * temp_R_aug[m, :],
            )

            temp_Q[:, [j, m]] = temp_Q[:, [j, m]] @ np.array([[c, -s], [s, c]])

        residual = temp_R_aug[m, n] if m >= n else 0.0

        if abs(residual) > self.TOL:
            # Inconsistency found, return the conflicting set
            conflict_vector = temp_Q.T[m, :]
            # The new constraint (at index m+1) is part of the conflict
            conflicting_indices = [
                k for k, w in enumerate(conflict_vector, 1) if abs(w) > self.TOL
            ]
            return conflicting_indices

        # Consistent: commit changes and return empty list
        self.Q = temp_Q
        self.R_aug = temp_R_aug
        self._constraints_count += 1
        return []


def generate_and_check_sparse_system(num_vars, num_constraints, density, num_conflicts):
    """
    Generates a large sparse system, introduces conflicts, and reports them.

    Args:
        num_vars (int): Number of variables (columns).
        num_constraints (int): Number of initial consistent constraints (rows).
        density (float): The sparsity of the constraint matrix A.
        num_conflicts (int): The number of synthetic conflicting constraints to create.
    """
    print(
        f"Setting up a system with {num_constraints} constraints, {num_vars} variables, and {num_conflicts} conflicts..."
    )

    # 1. Generate a sparse, consistent base system (Ax = b)
    print("--> Step 1: Generating a large, sparse, consistent system...")
    # Create a sparse matrix A
    A_sparse = scipy.sparse.random(
        num_constraints, num_vars, density=density, format="csr", random_state=42
    )
    # Create a "true" solution vector x_true
    x_true = np.random.randn(num_vars)
    # Calculate the consistent right-hand side b
    b_consistent = A_sparse @ x_true

    # Store all constraints (a_row, b_val) in a list
    all_constraints = [
        (A_sparse[i].toarray()[0], b_consistent[i]) for i in range(num_constraints)
    ]

    # 2. Generate synthetic conflicting constraints
    print(f"--> Step 2: Creating {num_conflicts} synthetic conflicting constraints...")
    for _ in range(num_conflicts):
        # Pick 2 to 5 random existing constraints to combine
        num_to_combine = random.randint(2, 5)
        indices_to_combine = random.sample(range(num_constraints), num_to_combine)
        weights = np.random.randn(num_to_combine)

        # Create a linear combination
        a_new = np.zeros(num_vars)
        b_new_consistent = 0.0
        for i, idx in enumerate(indices_to_combine):
            a_row, b_val = all_constraints[idx]
            a_new += weights[i] * a_row
            b_new_consistent += weights[i] * b_val

        # Perturb the b-value to create an inconsistency
        perturbation = (random.random() - 0.5) * 10
        b_new_inconsistent = b_new_consistent + perturbation

        all_constraints.append((a_new, b_new_inconsistent))

    # 3. Shuffle and process all constraints
    print(
        f"--> Step 3: Shuffling and processing all {len(all_constraints)} constraints..."
    )
    random.shuffle(all_constraints)

    manager = SilentConstraintManager(n_vars=num_vars)
    conflict_report = []
    start_time = time.time()

    for i, (a_row, b_val) in enumerate(all_constraints):
        # The manager only knows about the consistent constraints it has accepted.
        # Its internal count is `manager._constraints_count`.
        # The conflict list it returns is 1-based relative to its internal state.
        conflict_list = manager.add_constraint(a_row, b_val)

        is_consistent = not bool(conflict_list)

        if not is_consistent:
            # The new constraint (index i+1 in the shuffled list) is the cause
            conflict_list.append(manager._constraints_count + 1)

        report_entry = {
            "constraint_index": i + 1,
            "is_consistent_with_system": is_consistent,
            "conflicting_set": sorted(conflict_list) if not is_consistent else [],
        }
        conflict_report.append(report_entry)

    end_time = time.time()
    print(f"\nProcessing complete in {end_time - start_time:.2f} seconds.")
    return conflict_report


if __name__ == "__main__":
    # System parameters
    NUM_VARIABLES = 2000
    NUM_INITIAL_CONSTRAINTS = 2000
    DENSITY = 0.004
    NUM_CONFLICTS_TO_ADD = 50

    # Run the process
    final_report = generate_and_check_sparse_system(
        num_vars=NUM_VARIABLES,
        num_constraints=NUM_INITIAL_CONSTRAINTS,
        density=DENSITY,
        num_conflicts=NUM_CONFLICTS_TO_ADD,
    )

    # Print the summary of conflicts
    print("\n--- Conflict Summary ---")
    conflicts_found = 0
    for report in final_report:
        if not report["is_consistent_with_system"]:
            conflicts_found += 1
            idx = report["constraint_index"]
            conflicting_set = report["conflicting_set"]
            print(
                f"Constraint {idx} was INCONSISTENT. Conflicting set indices: {conflicting_set}"
            )

    if conflicts_found == 0:
        print("No inconsistencies were detected.")
    else:
        print(f"\nTotal conflicts detected: {conflicts_found}")
