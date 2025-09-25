import numpy as np
import scipy
from scipy import sparse
import time
from typing import List, Tuple
import pandas as pd

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

def L2_find(A, b, rel_threshold=0.25):
    """
    Minimizes L2 Norm by using least squares to identify inconsistent constraints in Ax = b
    same as QR? idt so 
    """

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    per_row_residuals = np.abs(A @ x - b)
    
    if per_row_residuals.size == 0:
        return []
    
    max_residual = np.max(per_row_residuals)
    if max_residual < 1e-10: 
        return []
        
    mask = per_row_residuals >= rel_threshold * max_residual
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
    
    solvers = [("L1", L1_find), ("L2", L2_find)]
    results, average = benchmark(cases, solvers)
    
    # for key, value in average.items():
    #     if key not in complete_averages:
    #         complete_averages[key] = 0
    #     complete_averages[key] += value
    
    df = pd.DataFrame(results)
    print(df)
    
    df = pd.DataFrame(average, index=[0])
    print(df)
    
    # print("\nOverall Averages:")
    # for key, value in complete_averages.items():
    #     print(f"{key}: {value / (len(cases) * id)}")





    





