import numpy as np
import scipy
from scipy import sparse
import time
from typing import List, Tuple
from ortools.linear_solver import pywraplp
import pandas as pd
import cvxopt

epsilon = 1e-8

def L1_find(A, b, rel_threshold=1-3e-2):
    """
    Uses L1 minimization to identify inconsistent constraints in Ax = b
    Minimize ||Ax - b||_1
    use dual variables, complementary slackness
    """
    A = np.asarray(A)
    m, n = A.shape
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    I = np.eye(m)
    A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n
    bounds_r = [(0, None)] * (2 * m)
    bounds = bounds_x + bounds_r
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs-ds')
    if not res.success:
        return None
    
    duals = np.asarray(res.eqlin.marginals, dtype=float)
    
    if duals is None or duals.size == 0:
        return []
    max_abs = np.max(np.abs(duals))
    if max_abs == 0:
        return []
    mask = np.abs(duals) >= rel_threshold * max_abs
    return [i for i, flag in enumerate(mask) if flag]

def L1_find_ortools(A, b, rel_threshold=1 - 3e-2):
    """
    Uses Google OR-Tools (GLOP solver) to find inconsistent constraints.
    This is a high-performance open-source option.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    m, n = A.shape

    # 1. Create the solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print("GLOP solver not available.")
        return None

    # 2. Create variables (x, r_pos, r_neg)
    infinity = solver.infinity()
    x = [solver.NumVar(-infinity, infinity, f'x_{i}') for i in range(n)]
    r_pos = [solver.NumVar(0, infinity, f'rp_{i}') for i in range(m)]
    r_neg = [solver.NumVar(0, infinity, f'rn_{i}') for i in range(m)]

    # 3. Define equality constraints: Ax - r_pos + r_neg = b
    constraints = []
    for i in range(m):
        expr = solver.Sum([A[i, j] * x[j] for j in range(n)])
        constraint = solver.Add(expr - r_pos[i] + r_neg[i] == b[i])
        constraints.append(constraint)

    # 4. Define the objective: minimize sum(r_pos + r_neg)
    objective = solver.Objective()
    for i in range(m):
        objective.SetCoefficient(r_pos[i], 1)
        objective.SetCoefficient(r_neg[i], 1)
    objective.SetMinimization()

    # 5. Solve the LP
    status = solver.Solve()

    # 6. Extract and process duals
    if status != pywraplp.Solver.OPTIMAL:
        print("OR-Tools solver did not find an optimal solution.")
        return None
        
    duals = np.array([c.dual_value() for c in constraints])
    
    if duals.size == 0: return []
    max_abs = np.max(np.abs(duals))
    if max_abs == 0: return []
    mask = np.abs(duals) >= rel_threshold * max_abs
    return [i for i, flag in enumerate(mask) if flag]

def benchmark(test_cases: List[Tuple[str, np.ndarray, np.ndarray, float]], solvers=List[Tuple[str, callable]]) -> List[dict]:
    """
    Also returns the average of each function over all test cases
    """
    averages = {}
    rows = []
    for solver, func in solvers:
        averages[f"{solver}_time"] = 0.0
        averages[f"{solver}_score"] = 0.0
        averages[f"{solver}_dense_time"] = 0.0
        averages[f"{solver}_sparse_time"] = 0.0

    for name, A, b, answer in test_cases:
        row = {"case": name, "m": A.shape[0], "n": A.shape[1], "bad": answer}
        for solver_name, solver in solvers:
            start_time = time.time()
            inconsistent = solver(A, b)
            elapsed = time.time() - start_time
            row[f"{solver_name}_time"] = elapsed
            row[f"{solver_name}_count"] = len(inconsistent) if inconsistent is not None else -1
            averages[f"{solver_name}_time"] += elapsed
            
            if "sparse" in name:
                averages[f"{solver_name}_sparse_time"] += elapsed
            else:
                averages[f"{solver_name}_dense_time"] += elapsed

            score = (1 - abs(len(inconsistent) - answer)/len(inconsistent))*100 if inconsistent is not None else 0
            row[f"{solver_name}_score"] = score
            averages[f"{solver_name}_score"] += score

        rows.append(row)
    
    for solver, _ in solvers:
        averages[f"{solver}_time"] /= len(test_cases)
        averages[f"{solver}_score"] /= len(test_cases)
        averages[f"{solver}_dense_time"] /= len(test_cases)
        averages[f"{solver}_sparse_time"] /= len(test_cases)

    return rows, averages

def create_infeasible_system(m: int, n: int, bad : int):
    """
    Creates an m x n system of linear equations with
    bad infeasible constraints.
    """

    true_solution_x = np.random.randn(n, 1)
    A = np.random.randn(m, n)
    b = A @ true_solution_x

    num_consistent_base = max(1, n - bad)
    

    for i in range(m - bad, m):
        combination_coeffs = np.random.randn(1, num_consistent_base)
        new_A_row = combination_coeffs @ A[:num_consistent_base, :]
        
        consistent_b_value = combination_coeffs @ b[:num_consistent_base]
        inconsistent_b_value = consistent_b_value + (np.random.rand() + 0.5)
        A[i, :] = new_A_row
        b[i] = inconsistent_b_value
        
    return A, b

def create_sparse_infeasible_system(m: int, n: int, bad : int, density: float = 0.1):
    """
    Creates an m x n system of sparse linear equations with
    bad infeasible constraints.
    """

    true_solution_x = np.random.randn(n, 1)
    A = sparse.random(m, n, density=density, format="lil",data_rvs=np.random.randn)
    b = A @ true_solution_x
    num_consistent_base = max(1, n - bad)
    
    for i in range(m - bad, m):
        combination_coeffs = np.random.randn(1, num_consistent_base)
        new_A_row = combination_coeffs @ A[:num_consistent_base, :]
        

        consistent_b_value = combination_coeffs @ b[:num_consistent_base]
        inconsistent_b_value = consistent_b_value + (np.random.rand() + 0.5)

        A[i, :] = new_A_row
        b[i] = inconsistent_b_value
        
    return A.toarray(), b

def find_inconsistent_svd(A, b, rel_threshold=1 - 3e-2, rank_tol=1e-10):
    """
    Identifies inconsistent constraints in Ax = b using SVD.

    This function finds the direction in the left null space of A that has the
    largest projection onto b. The components of this direction vector are
    treated as weights, and the constraints with the largest weights are
    identified as the most inconsistent.

    Args:
        A (np.ndarray): The input matrix.
        b (np.ndarray): The corresponding vector.
        rel_threshold (float): The relative threshold for selecting the most
                               involved constraints.
        rank_tol (float): The tolerance for identifying near-zero singular values.

    Returns:
        list: A list of integer indices for the identified inconsistent constraints.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    m, n = A.shape

    # 1. Compute the full SVD to get the complete U matrix.
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    # 2. Determine the rank and identify the left null space.
    # The left null space is spanned by the columns of U corresponding
    # to singular values that are effectively zero.
    rank = np.sum(s > rank_tol)
    
    # If the matrix has full row rank, there's no row dependency to detect.
    if rank == m:
        return []

    left_null_vectors = U[:, rank:]

    # 3. Find the direction of maximum inconsistency.
    # This is the vector in the left null space that b has the largest
    # projection onto.
    inconsistency_scores = np.abs(left_null_vectors.T @ b)
    
    if np.all(inconsistency_scores < rank_tol):
        # The system is consistent
        return []
    
    best_index = np.argmax(inconsistency_scores)
    u_culprit = left_null_vectors[:, best_index]

    # 4. Apply the heuristic to the components of the "culprit" vector.
    # The components of u_culprit are analogous to the dual variables in the
    # L1 method. They are the weights of the inconsistent row combination.
    max_abs_component = np.max(np.abs(u_culprit))

    if max_abs_component < rank_tol:
        return []

    mask = np.abs(u_culprit) >= rel_threshold * max_abs_component
    return [i for i, flag in enumerate(mask) if flag]





#tests
if __name__ == "__main__":

    sizes = [(1000, 1000)]
    NUM = 2
    complete_averages = {}
    bads = [20, 50, 80, 110] 
    id = 0
    cases = []
    for bad in bads:
        for _ in range(NUM):
            for (m, n) in sizes:
                A, b = create_infeasible_system(n, m, bad)
                cases.append((f"rand_{m}x{n}_seed{id}", A, b, bad))
                A, b = create_sparse_infeasible_system(m, n, bad)
                cases.append((f"sparse_{m}x{n}_seed{id}", A, b, bad))
                id += 1
    
    solvers = [("SVD", find_inconsistent_svd), ("L1", L1_find)]
    results, average = benchmark(cases, solvers)

    
    df = pd.DataFrame(results)
    print(df)
    
    df = pd.DataFrame(average, index=[0])
    print(df)
    



    





