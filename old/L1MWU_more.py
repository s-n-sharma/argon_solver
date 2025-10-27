import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog
import time
from typing import List, Tuple, Optional, Callable, Dict
import pandas as pd

# ---------------- Core Helper Functions ---------------- #

def l1_dual_via_linprog(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Helper: Solves the L1 LP once to return the equality duals."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    I = np.eye(m)
    A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n
    bounds_r = [(0, None)] * (2 * m)
    bounds = bounds_x + bounds_r
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs-ds')
    if not res.success:
        return None
    return np.asarray(res.eqlin.marginals, dtype=float)

def l1_error(A: np.ndarray, b: np.ndarray) -> float:
    """Helper: Returns the minimum L1 error ||Ax - b||_1 for a system."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    if m == 0:
        return 0.0
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    I = np.eye(m)
    A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n
    bounds_r = [(0, None)] * (2 * m)
    bounds = bounds_x + bounds_r
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs-ds')
    return float(res.fun) if res.success else float('inf')

def _thin_colspace_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Helper: Returns a thin orthonormal basis Q for col(A) using SVD."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0: return np.zeros((A.shape[0], 0))
    smax = s[0]
    r = int(np.sum(s > max(tol, 1e-12) * smax))
    return U[:, :r] if r > 0 else np.zeros((A.shape[0], 0))

def mwu_dual_solve(A: np.ndarray, b: np.ndarray, steps: int = 250, eta: Optional[float] = None) -> np.ndarray:
    """Helper: Approximates the L1 dual vector y via fast projected gradient ascent."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    Q = _thin_colspace_basis(A)
    def proj_null(yv: np.ndarray) -> np.ndarray:
        return yv if Q.size == 0 else yv - Q @ (Q.T @ yv)
    y = np.zeros(m)
    if eta is None: eta = 0.9 / (np.linalg.norm(b) + 1e-9)
    for _ in range(1, steps + 1):
        y += eta * b
        y = proj_null(y)
        np.clip(y, -1.0, 1.0, out=y)
    return y

# ---------------- Constraint Finding Solvers ---------------- #

def L1_find(A, b, rel_threshold=1 - 3e-2):
    """Original Method: Uses a simple relative threshold on L1 duals."""
    duals = l1_dual_via_linprog(A, b)
    if duals is None or duals.size == 0: return []
    max_abs = np.max(np.abs(duals))
    if max_abs == 0: return []
    mask = np.abs(duals) >= rel_threshold * max_abs
    return [i for i, flag in enumerate(mask) if flag]

def L2_find(A, b, rel_threshold=0.25):
    """L2 Method: Uses least-squares residuals to find outliers."""
    x, _, _, _ = np.linalg.lstsq(A, b.reshape(-1), rcond=None)
    per_row_residuals = np.abs(A @ x - b.reshape(-1))
    if per_row_residuals.size == 0: return []
    max_residual = np.max(per_row_residuals)
    if max_residual < 1e-10: return []
    mask = per_row_residuals >= rel_threshold * max_residual
    return [i for i, flag in enumerate(mask) if flag]

def MWU_find(A: np.ndarray, b: np.ndarray, rel_threshold: float = 0.99, steps: int = 300) -> List[int]:
    """Fast Method: Uses an approximate dual solver with a relative threshold."""
    y = mwu_dual_solve(A, b, steps=steps)
    if y.size == 0: return []
    max_abs = float(np.max(np.abs(y)))
    if max_abs <= 1e-15: return []
    mask = np.abs(y) >= rel_threshold * max_abs
    return [i for i, flag in enumerate(mask) if flag]

def MMWU_find(A: np.ndarray, b: np.ndarray, rel_threshold: float = 0.99, steps: int = 200, trials: int = 50, z_score : float = 1) -> List[int]:
    """Multi-Trial Fast Method: Runs MWU multiple times and aggregates results."""
    m = A.shape[0]
    hit_counts = np.zeros(m, dtype=int)
    for _ in range(trials):
        candidates = MWU_find(A, b, rel_threshold=rel_threshold, steps=steps)
        for i in candidates:
            hit_counts[i] += 1
        
    hit_count_std = np.std(hit_counts)
    hit_count_mean = np.mean(hit_counts)
    final_hits = np.array([i for i, count in enumerate(hit_counts) if count >= (hit_count_mean + z_score * hit_count_std)])
    return final_hits

def Hybrid_find(A: np.ndarray, b: np.ndarray, rel_threshold: float = 0.99, tol: float = 1e-8, steps: int = 200) -> List[int]:
    """Hybrid Method: Uses MWU to find candidates, then confirms with precise L1 error checks."""
    candidates = MWU_find(A, b, rel_threshold=rel_threshold, steps=steps)
    if not candidates: return []
    confirmed_hits: List[int] = []
    for i in candidates:
        A_tmp = np.delete(A, i, axis=0)
        b_tmp = np.delete(b, i)
        if l1_error(A_tmp, b_tmp) < tol:
            confirmed_hits.append(i)
    return confirmed_hits

def L1_find_iterative(A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> List[int]:
    """Robust Method: Iteratively finds and removes the single worst offender."""
    A_current = np.copy(A)
    b_current = np.copy(b)
    original_indices = list(range(A.shape[0]))
    inconsistent_indices = []
    while True:
        error = l1_error(A_current, b_current)
        if error < tol: break
        duals = l1_dual_via_linprog(A_current, b_current)
        if duals is None or duals.size == 0: break
        worst_local_idx = np.argmax(np.abs(duals))
        worst_original_idx = original_indices.pop(worst_local_idx)
        inconsistent_indices.append(worst_original_idx)
        A_current = np.delete(A_current, worst_local_idx, axis=0)
        b_current = np.delete(b_current, worst_local_idx, axis=0)
        if A_current.shape[0] == 0: break
    return sorted(inconsistent_indices)

def L1_find_statistical(A: np.ndarray, b: np.ndarray, k: float = 3.0) -> List[int]:
    """Improved One-Shot Method: Uses a statistical threshold on L1 duals."""
    duals = l1_dual_via_linprog(A, b)
    if duals is None or duals.size == 0: return []
    abs_duals = np.abs(duals)
    mean_abs = np.mean(abs_duals)
    std_abs = np.std(abs_duals)
    if std_abs < 1e-9: return []
    threshold = mean_abs + k * std_abs
    mask = abs_duals >= threshold
    return [i for i, flag in enumerate(mask) if flag]

def L1_GMWU(A: np.ndarray, b: np.ndarray,z_score: float = 1, steps: int = 200, trials: int = 100, rel_threshold: float = 1, tol: float = 1e-8) -> List[int]:
    """
    1. given A, normalize its rows to unit L2 norm and make it into a new matrix A_prime
    2. compute dot products between each row to see which are the most similar, compute sum of dot products for each row
    3. subtract from that sum the number of non_zero entries in that row
    4. initialize a weight vector w where weight of each row is proportional to exp(-alpha * adjusted_sum)
    5. run a linear program highs-ds that focuses on minimizing the weighted L1 error, using the weights w
    6. for each dual, if it is greater than z_score standard deviation above the mean, mark it as inconsistent, reduce weight
    7. perturb the weights randomly a bit
    8. repeat steps 5-7 for a number of trials
    9. return all rows with weights with z_score standard deviations below the mean
    """
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    if m == 0: return []
    
    row_norms = np.linalg.norm(A, axis=1, ord=2)
    row_norms[row_norms < 1e-12] = 1.0
    A_prime = A / row_norms[:, np.newaxis]
    
    similarity_matrix = A_prime @ A_prime.T
    np.fill_diagonal(similarity_matrix, 0)
    adjusted_sums = np.sum(similarity_matrix, axis=1) - np.count_nonzero(A, axis=1)
    
    alpha = 0.5 / (np.std(adjusted_sums) + 1e-9)
    weights = np.exp(-alpha * adjusted_sums)
    
    hit_counts = np.zeros(m, dtype=int)
    
    for _ in range(trials):
        #c = np.concatenate([weights * np.zeros(n), weights * np.ones(2 * m)])
        c = np.concatenate([np.zeros(n), np.hstack([weights, weights]) * np.ones(2 * m)])
        I = np.eye(m)
        A_eq = np.hstack([A, -I, I])
        bounds_x = [(None, None)] * n
        bounds_r = [(0, None)] * (2 * m)
        bounds = bounds_x + bounds_r
        
        res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs-ds')
        if not res.success:
            continue
        
        duals = np.asarray(res.eqlin.marginals, dtype=float)
        if duals is None or duals.size == 0:
            continue
        
        abs_duals = np.abs(duals)
        mean_abs = np.mean(abs_duals)
        std_abs = np.std(abs_duals)
        if std_abs < 1e-9:
            continue
        
        threshold = mean_abs + z_score * std_abs
        for i, val in enumerate(abs_duals):
            if val >= threshold:
                hit_counts[i] += 1
                weights[i] *= 0.9
        
        perturbation = np.random.normal(0, 0.05, size=m)
        weights += perturbation
        weights[weights < 1e-3] = 1e-3
    
    hit_count_std = np.std(hit_counts)
    print(hit_count_std)
    hit_count_mean = np.mean(hit_counts)

    final_hits = np.array([i for i, count in enumerate(hit_counts) if count >= (hit_count_mean + 3 * hit_count_std)])
    return final_hits

import sklearn
from sklearn.linear_model import LinearRegression

def calculate_influence_matrix(data: list[tuple[list[float], list[float]]]) -> np.ndarray:
    """
    Calculates the Influence Matrix using Multiple Linear Regression.

    The matrix M is N x N, where M[i, j] is the coefficient (influence)
    of the i-th constraint weight on the j-th constraint dual value.

    Args:
        data: A list of tuples, where each tuple is (weights_list, dual_values_list).
              e.g., [([w1, w2, w3], [d1, d2, d3]), (..., ...)]

    Returns:
        A pandas DataFrame representing the N x N Influence Matrix.
        Rows: Input Weights (Independent Variables)
        Columns: Output Dual Values (Dependent Variables)
    """
    if not data:
        raise ValueError("Input data cannot be empty.")

    # 1. Prepare Data
    # Separate weights (X) and dual values (Y)
    X_data = np.array([item[0] for item in data])
    Y_data = np.array([item[1] for item in data])

    N = X_data.shape[1]
    if Y_data.shape[1] != N:
        raise ValueError("The number of weights and dual values must be equal.")
    
    # Initialize the Influence Matrix (N rows x N columns)
    # Rows will be the weight indices, Columns will be the dual value indices
    influence_matrix = np.zeros((N, N))
    
    # Create labels for the resulting DataFrame
    constraint_labels = [f'Constraint {i+1}' for i in range(N)]

    # 2. Run N Separate Regressions
    # We run one regression for each dual value d_j, using ALL weights w_i
    for j in range(N):
        # The target variable y is the j-th dual value across all observations
        y_j = Y_data[:, j]
        
        # The features X are all the weights across all observations
        X = X_data
        
        # Initialize and fit the linear regression model
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y_j)
        
        # The coefficients (excluding the intercept) represent the influence
        # of each weight w_i on the dual value d_j.
        # This coefficient vector forms the j-th column of our matrix.
        influence_matrix[:, j] = model.coef_
    
    return influence_matrix

def L1_RCMWU(A: np.ndarray, b: np.ndarray, rel_threshold: float = 0.99, steps: int = 200, trials: int = 50, z_score : float = 1) -> List[int]:
    """
    1) initialize weights for all constraints to be uniform or 1
    2) randomly sample a subset of the constraints (rows of A and entries of b)
    3) increase the weights of the sampled constraints by a significant random factor
    4) run L1 LP with the current weights to get dual values, reset weights to 1
    5) repeat 3) and 4) for a number of iterations
    6) calculate influence matrix from weights to duals using the function calculate_influence_matrix
    7) calculate the std, mean of the entire influence matrix
    8) for non-diagonal elements, if an entry is greater than z_score standard deviations above the mean, mark its row and column as inconsistent
    9) return all rows and columns marked as inconsistent (use a set to avoid duplicates, and then convert to list)
    """
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    if m == 0: return []
    
    data_for_influence = []
    
    for _ in range(trials):
        weights = np.ones(m)
        sample_size = max(1, m // 10)
        sampled_indices = np.random.choice(m, size=sample_size, replace=False)
        for idx in sampled_indices:
            weights[idx] *= 100#(1.5 + np.random.rand())
        
        c = np.concatenate([np.zeros(n), np.hstack([weights, weights]) * np.ones(2 * m)])
        I = np.eye(m)
        A_eq = np.hstack([A, -I, I])
        bounds_x = [(None, None)] * n
        bounds_r = [(0, None)] * (2 * m)
        bounds = bounds_x + bounds_r
        
        res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs-ds')
        if not res.success:
            continue
        
        duals = np.asarray(res.eqlin.marginals, dtype=float)
        if duals is None or duals.size == 0:
            continue
        
        data_for_influence.append((weights.tolist(), duals.tolist()))
    
    if not data_for_influence:
        return []
    
    influence_matrix = calculate_influence_matrix(data_for_influence)
    
    mean_influence = np.mean(influence_matrix)
    std_influence = np.std(influence_matrix)

    print(mean_influence, std_influence)
    
    inconsistent_indices_set = set()
    
    for i in range(m):
        for j in range(m):
            if i != j and influence_matrix[i, j] > (mean_influence + 3 * std_influence):
                inconsistent_indices_set.add(i)
                inconsistent_indices_set.add(j)
    
    return sorted(list(inconsistent_indices_set))


# ---------------- Benchmark Suite ---------------- #

def create_infeasible_system(m: int, n: int, bad: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Creates a dense m x n system with 'bad' strongly inconsistent constraints."""
    true_solution_x = np.random.randn(n, 1)
    A = np.random.randn(m, n)
    b = A @ true_solution_x
    bad_indices = list(range(m - bad, m))
    num_consistent_base = max(1, n - bad)
    for i in bad_indices:
        coeffs = np.random.randn(1, num_consistent_base)
        new_A_row = coeffs @ A[:num_consistent_base, :]
        consistent_b = coeffs @ b[:num_consistent_base]
        inconsistent_b = consistent_b + (np.random.rand() + 0.5) * np.sign(np.random.randn())
        A[i, :] = new_A_row
        b[i] = inconsistent_b
    return A, b, bad_indices

def benchmark(test_cases: List[Tuple[str, np.ndarray, np.ndarray, List[int]]], solvers: List[Tuple[str, Callable]]) -> Tuple[List[dict], dict]:
    """Benchmarks solvers on test cases, using F1 score for accuracy."""
    def f1_measure(pred: List[int], truth: List[int]) -> float:
        pred_set, truth_set = set(pred), set(truth)
        if not truth_set and not pred_set: return 100.0
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return f1 * 100
    rows = []
    averages = {f"{s_name}_{m}": 0.0 for s_name, _ in solvers for m in ["time", "score"]}
    for name, A, b, bad_indices in test_cases:
        row = {"case": name, "m": A.shape[0], "n": A.shape[1], "truth_count": len(bad_indices)}
        for solver_name, solver_func in solvers:
            start_time = time.time()
            pred_indices = solver_func(A, b)
            elapsed = time.time() - start_time
            score = f1_measure(pred_indices if pred_indices is not None else [], bad_indices)
            row[f"{solver_name}_time"] = elapsed
            row[f"{solver_name}_pred_count"] = len(pred_indices) if pred_indices is not None else -1
            row[f"{solver_name}_f1_score"] = score
            averages[f"{solver_name}_time"] += elapsed
            averages[f"{solver_name}_score"] += score
        rows.append(row)
    for key in averages:
        averages[key] /= len(test_cases)
    return rows, averages

def create_sizes(m):
    """Helper: Generates a list of (m, n) sizes for test cases."""
    assert m > 300
    step_size = 300
    sizes = []
    for i in range(2, (m // step_size) + 1):
        sizes.append((m, i * step_size))
    
    return sizes



# ---------------- Main Execution Block ---------------- #

if __name__ == "__main__":
    pd.set_option('display.width', 150)
    pd.set_option('display.precision', 4)

    # Setup for the "hard" test case: overdetermined system with strong, conflicting noise
    sizes = [(300, 90), (300, 120), (300, 210), (300, 300)] 
    #sizes = create_sizes(500)
    NUM_RUNS_PER_SETTING = 1
    bads = [90] 
    cases = []
    
    print("Generating test cases...")
    for m, n in sizes:
        for bad_count in bads:
            for i in range(NUM_RUNS_PER_SETTING):
                A, b, bad_idx = create_infeasible_system(m, n, bad_count)
                cases.append((f"dense_{m}x{n}_bad{bad_count}_run{i+1}", A, b, bad_idx))

    print(f"Benchmarking {len(cases)} cases...")
    solvers = [
        ("L1", L1_find), 
        ("L2", L2_find),
        ("MWU", MWU_find),
        #("MMWU", MMWU_find),
        #("GMWU", L1_GMWU),
        ("RCMWU", L1_RCMWU),
        ("Hybrid", Hybrid_find),
        #("L1_Statistical", L1_find_statistical),
        #("L1_Iterative", L1_find_iterative)
    ]
    results, average = benchmark(cases, solvers)
    
    print("\n--- Individual Case Results ---")
    df_results = pd.DataFrame(results)
    print(df_results)
    
    print("\n\n--- Average Performance ---")
    df_average = pd.DataFrame(average, index=[0])
    print(df_average)