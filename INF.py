import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog
import time
from typing import List, Tuple, Optional, Callable, Dict
import pandas as pd

# ---------------- Core Helper Functions ---------------- #

def l1_dual_via_linprog_weighted(A: np.ndarray, b: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    """Helper: Solves a WEIGHTED L1 LP to return the duals."""
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)
    m, n = A.shape
    c = np.concatenate([np.zeros(n), weights, weights]) # Use weights in the objective
    I = np.eye(m)
    A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n
    bounds_r = [(0, None)] * (2 * m)
    bounds = bounds_x + bounds_r
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs')
    if not res.success: return None
    return np.asarray(res.eqlin.marginals, dtype=float)

def l1_dual_via_linprog(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Helper: Solves the standard L1 LP once to return the equality duals."""
    weights = np.ones(A.shape[0])
    return l1_dual_via_linprog_weighted(A, b, weights)

def l1_error(A: np.ndarray, b: np.ndarray) -> float:
    """Helper: Returns the minimum L1 error ||Ax - b||_1 for a system."""
    A = np.asarray(A); b = np.asarray(b).reshape(-1); m, n = A.shape
    if m == 0: return 0.0
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    I = np.eye(m); A_eq = np.hstack([A, -I, I])
    bounds_x = [(None, None)] * n; bounds_r = [(0, None)] * (2 * m)
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds_x + bounds_r, method='highs')
    return float(res.fun) if res.success else float('inf')

def _thin_colspace_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Helper: Returns a thin orthonormal basis Q for col(A) using SVD."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0: return np.zeros((A.shape[0], 0))
    r = int(np.sum(s > max(tol, 1e-12) * s[0]))
    return U[:, :r] if r > 0 else np.zeros((A.shape[0], 0))

def mwu_dual_solve(A: np.ndarray, b: np.ndarray, steps: int = 120, eta: Optional[float] = None) -> np.ndarray:
    """Helper: Approximates the L1 dual vector y via fast projected gradient ascent."""
    A = np.asarray(A); b = np.asarray(b).reshape(-1); m, n = A.shape
    Q = _thin_colspace_basis(A)
    def proj_null(yv: np.ndarray) -> np.ndarray: return yv if Q.size == 0 else yv - Q @ (Q.T @ yv)
    y = np.zeros(m)
    if eta is None: eta = 0.9 / (np.linalg.norm(b) + 1e-9)
    for _ in range(1, steps + 1):
        y += eta * b; y = proj_null(y); np.clip(y, -1.0, 1.0, out=y)
    return y

def mwu_dual_solve_weighted(A: np.ndarray, b: np.ndarray, weights: np.ndarray, steps: int = 80) -> np.ndarray:
    """Helper: Approximates the duals of a WEIGHTED L1 problem."""
    A = np.asarray(A); b = np.asarray(b).reshape(-1); m, n = A.shape
    Q = _thin_colspace_basis(A)
    def proj_null(yv: np.ndarray) -> np.ndarray: return yv if Q.size == 0 else yv - Q @ (Q.T @ yv)
    y = np.zeros(m)
    eta = 0.9 / (np.linalg.norm(b) + 1e-9)
    w_clipped = np.asarray(weights).reshape(-1)
    
    for _ in range(1, steps + 1):
        y += eta * b
        y = proj_null(y)
        np.clip(y, -w_clipped, w_clipped, out=y)
    return y

# ---------------- Constraint Finding Solvers ---------------- #

def L1_find(A, b, rel_threshold=1 - 3e-2):
    """Method 1: Simple relative threshold on L1 duals."""
    duals = l1_dual_via_linprog(A, b)
    if duals is None or duals.size == 0: return []
    max_abs = np.max(np.abs(duals));
    if max_abs == 0: return []
    return [i for i, v in enumerate(np.abs(duals)) if v >= rel_threshold * max_abs]

def L2_find(A, b, rel_threshold=0.25):
    """Method 2: Simple relative threshold on L2 least-squares residuals."""
    x, _, _, _ = np.linalg.lstsq(A, b.reshape(-1), rcond=None)
    res = np.abs(A @ x - b.reshape(-1))
    if res.size == 0: return []
    max_res = np.max(res)
    if max_res < 1e-10: return []
    return [i for i, v in enumerate(res) if v >= rel_threshold * max_res]

def MWU_find(A: np.ndarray, b: np.ndarray, rel_threshold: float = 0.99, steps: int = 300) -> List[int]:
    """Method 3: Fast approximate duals with a relative threshold."""
    y = mwu_dual_solve(A, b, steps=steps)
    if y.size == 0: return []
    max_abs = float(np.max(np.abs(y)))
    if max_abs <= 1e-15: return []
    return [i for i, v in enumerate(np.abs(y)) if v >= rel_threshold * max_abs]

def L1_find_iterative(A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> List[int]:
    """Method 4: Robust but slow 'peeling' method; iteratively finds and removes the worst offender."""
    A_curr, b_curr = np.copy(A), np.copy(b)
    orig_indices, bad_indices = list(range(A.shape[0])), []
    while True:
        if l1_error(A_curr, b_curr) < tol: break
        duals = l1_dual_via_linprog(A_curr, b_curr)
        if duals is None or duals.size == 0: break
        worst_idx = np.argmax(np.abs(duals))
        bad_indices.append(orig_indices.pop(worst_idx))
        A_curr = np.delete(A_curr, worst_idx, axis=0)
        b_curr = np.delete(b_curr, worst_idx, axis=0)
        if A_curr.shape[0] == 0: break
    return sorted(bad_indices)

def L1_find_influence_fast(A: np.ndarray, b: np.ndarray, num_candidates: int = 30, probe_weight: float = 1000.0) -> List[int]:
    """Method 5: Fast and accurate influence method using an approximate MWU probe."""
    m = A.shape[0]
    if m == 0: return []

    num_candidates = max(num_candidates, m//3)
    
    # Step 1: Better Candidate Filtering using MWU
    candidates = MWU_find(A, b, rel_threshold=0.5, steps=500)
    if not candidates: candidates = L2_find(A, b, rel_threshold=0.1) # Fallback
    if len(candidates) > num_candidates:
        candidates = candidates[:num_candidates]
    if not candidates: return []

    influence_scores = np.zeros(m)
    
    # Step 2: Probing loop
    for i in candidates:
        weights = np.ones(m)
        weights[i] = probe_weight # Set the probe weight
        
        # Step 3: Fast Approximate Probing using weighted MWU
        duals = mwu_dual_solve_weighted(A, b, weights)
        
        # Step 4: Improved Influence Scoring using dual magnitude
        duals[i] = 0 # Ignore self-influence
        influence_scores += np.abs(duals)

    if not np.any(influence_scores): return []
    
    # Step 5: Robust Final Selection using statistical outliers
    mean_score = np.mean(influence_scores[influence_scores > 0])
    std_score = np.std(influence_scores[influence_scores > 0])
    if std_score < 1e-9: return candidates # Fallback if variance is zero
    
    threshold = mean_score + 1 * std_score
    return [i for i, score in enumerate(influence_scores) if score > threshold]

def L1_find_adaptive_weights(
    A: np.ndarray, 
    b: np.ndarray, 
    iterations: int = 100, 
    learning_rate: float = 0.5,
    conv_tol: float = 1e-4
) -> List[int]:
    """
    Finds inconsistent constraints using the adaptive weighting scheme.
    """
    m, n = A.shape
    if m == 0: return []
    
    weights = np.ones(m)

    for i in range(iterations):
        weights_old = weights.copy()
        
        # Solve the weighted L1 problem to get the current stress (duals)
        duals = l1_dual_via_linprog_weighted(A, b, weights)
        
        if duals is None:
            # Solver failed, return based on current weights if possible
            break

        # Multiplicative exponential update rule:
        # Penalize constraints with high duals by reducing their weight
        update_factors = np.exp(-learning_rate * np.abs(duals))
        weights = weights * update_factors
        
        # Re-normalize weights to maintain a constant sum and prevent decay to zero
        sum_w = np.sum(weights)
        if sum_w < 1e-9: break # Avoid division by zero
        weights = m * weights / sum_w
        
        # Check for convergence
        if np.linalg.norm(weights - weights_old) < conv_tol:
            # print(f"Converged at iteration {i+1}")
            break
            
    # Final selection: The culprits are the constraints whose weights have been driven low
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    if std_w < 1e-9: return [] # No significant difference in weights

    # Identify outliers on the low end of the weight distribution
    threshold = mean_w - 1.5 * std_w
    return [i for i, w in enumerate(weights) if w < threshold]


def MWU_find_adaptive_weights(
    A: np.ndarray, 
    b: np.ndarray, 
    iterations: int = 100, 
    learning_rate: float = 1.0, # Note: a higher learning rate may work better for MWU
    conv_tol: float = 1e-4
) -> List[int]:
    """
    Finds inconsistent constraints using a fast, fully-approximate adaptive weighting scheme.
    """
    m, n = A.shape
    if m == 0: return []
    
    weights = np.ones(m)

    for i in range(iterations):
        weights_old = weights.copy()
        
        # KEY CHANGE: Use the fast MWU solver instead of the slow linprog solver
        duals = mwu_dual_solve_weighted(A, b, weights)
        
        # Multiplicative exponential update rule
        update_factors = np.exp(-learning_rate * np.abs(duals))
        weights = weights * update_factors
        
        # Re-normalize weights
        sum_w = np.sum(weights)
        if sum_w < 1e-9: break
        weights = m * weights / sum_w
        
        # Check for convergence
        if np.linalg.norm(weights - weights_old) < conv_tol:
            # print(f"Converged at iteration {i+1}")
            break
            
    # Final selection: The culprits are the constraints whose weights have been driven low
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    if std_w < 1e-9: return []

    threshold = mean_w - 1.5 * std_w
    return [i for i, w in enumerate(weights) if w < threshold]

def MWU_find_correlation_weights(
    A: np.ndarray, 
    b: np.ndarray, 
    outer_iterations: int = 20, 
    learning_rate: float = 0.1,
    num_candidates: int = 25,
    probe_weight: float = 50.0
) -> List[int]:
    """
    Finds inconsistent constraints using correlation-based adaptive weighting.
    """
    m, n = A.shape
    if m == 0: return []

    # 1. Calculate baseline duals to measure changes against
    duals_base = mwu_dual_solve(A, b)

    # 2. Select a smaller set of candidates to probe
    candidates = MWU_find(A, b, rel_threshold=0.2)
    if not candidates or len(candidates) < 2: return [] # Method requires candidates
    if len(candidates) > num_candidates:
        candidates = candidates[:num_candidates]

    # This is the main weight vector we will adapt over time
    main_weights = np.ones(m)

    # 3. Outer Loop: Adapt the main weights
    for _ in range(outer_iterations):
        total_update_direction = np.zeros(m)

        # 4. Inner Loop: Probe candidates to find correlations
        for i in candidates:
            probe_weights = main_weights.copy()
            probe_weights[i] = probe_weight
            
            duals_probe = mwu_dual_solve_weighted(A, b, probe_weights)

            # 5. Correlation Analysis
            # Compare the sign of the probed dual vs. the baseline dual
            # A positive correlation means signs match and magnitude increased
            # A negative correlation means signs mismatch
            signs_base = np.sign(duals_base)
            signs_probe = np.sign(duals_probe)
            
            # Increase weight for negatively correlated, decrease for positively correlated
            # A simple update direction rule:
            # -1 for positive correlation (decrease weight)
            # +1 for negative correlation (increase weight)
            update_direction = -1 * (signs_base * signs_probe)
            total_update_direction += update_direction

        # 6. Update the main weights based on the summed influences from all probes
        update_factors = np.exp(learning_rate * total_update_direction / len(candidates))
        main_weights = main_weights * update_factors
        
        # Re-normalize
        main_weights = m * main_weights / np.sum(main_weights)

    # 7. Final Selection: Culprits have the lowest final weights
    mean_w = np.mean(main_weights)
    std_w = np.std(main_weights)
    if std_w < 1e-9: return []

    threshold = mean_w - 1.5 * std_w
    return [i for i, w in enumerate(main_weights) if w < threshold]

def MWU_find_gradient_weights(
    A: np.ndarray, 
    b: np.ndarray, 
    outer_iterations: int = 20, 
    learning_rate: float = 0.05, # A smaller learning rate is often needed for gradient methods
    num_candidates: int = 25,
    probe_weight: float = 50.0
) -> List[int]:
    """
    Finds inconsistent constraints using gradient-based adaptive weighting.
    The 'gradient' is the change in duals after probing.
    """
    m, n = A.shape
    if m == 0: return []

    num_candidates = max(num_candidates, m//3)

    # 1. Calculate baseline duals to measure changes against
    duals_base = mwu_dual_solve(A, b)

    # 2. Select candidates to probe
    candidates = MWU_find(A, b, rel_threshold=0.2)
    if not candidates or len(candidates) < 2: return []
    if len(candidates) > num_candidates:
        candidates = candidates[:num_candidates]

    main_weights = np.ones(m)

    # 3. Outer Loop: Adapt the main weights
    for _ in range(outer_iterations):
        total_dual_change = np.zeros(m)

        # 4. Inner Loop: Probe candidates to find the gradient of the duals
        for i in candidates:
            probe_weights = main_weights.copy()
            probe_weights[i] = probe_weight
            
            duals_probe = mwu_dual_solve_weighted(A, b, probe_weights)

            # 5. Gradient Calculation
            # The change in duals represents the influence or 'gradient'
            dual_change = duals_probe - duals_base
            total_dual_change += dual_change

        # 6. Update main weights based on the average change
        # A positive change means the constraint fought the probe (increase weight)
        # A negative change means it agreed with the probe (decrease weight)
        avg_dual_change = total_dual_change / len(candidates)
        
        # We scale the change by the learning rate and apply it multiplicatively
        update_factors = np.exp(learning_rate * avg_dual_change)
        main_weights = main_weights * update_factors
        
        # Re-normalize
        main_weights = m * main_weights / np.sum(main_weights)

    # 7. Final Selection: Culprits have the lowest final weights
    mean_w = np.mean(main_weights)
    std_w = np.std(main_weights)
    if std_w < 1e-9: return []

    threshold = mean_w - 1.5 * std_w
    return [i for i, w in enumerate(main_weights) if w < threshold]

def MWU_find_gradient_regularized(
    A: np.ndarray, 
    b: np.ndarray, 
    outer_iterations: int = 30, # May need more iterations for biases to take effect
    learning_rate: float = 0.05,
    num_candidates: int = 25,
    probe_weight: float = 50.0,
    simplicity_bias_factor: float = 1.01, # Factor > 1 to favor simpler solutions
    memory_decay: float = 0.7 # How fast to "forget" past dominant constraints
) -> List[int]:
    """
    Finds inconsistent constraints using a regularized, gradient-based adaptive 
    weighting scheme with biases for simplicity and diversity.
    """
    m, n = A.shape
    if m == 0: return []

    # 1. Calculate baseline duals to measure changes against
    duals_base = mwu_dual_solve(A, b)
    num_candidates = max(num_candidates, m//3)

    # 2. Select candidates to probe
    candidates = MWU_find(A, b, rel_threshold=0.2)
    if not candidates or len(candidates) < 2: return []
    if len(candidates) > num_candidates:
        candidates = candidates[:num_candidates]

    main_weights = np.ones(m)
    influence_memory = np.zeros(m) # Memory for the diversity bias

    # 3. Outer Loop: Adapt the main weights
    for _ in range(outer_iterations):
        total_dual_change = np.zeros(m)

        # Apply the diversity bias based on memory from PREVIOUS iterations
        # This penalizes consistently dominant constraints to encourage exploration.
        penalty_factors = np.exp(-learning_rate * influence_memory)
        main_weights = main_weights * penalty_factors
        
        # 4. Inner Loop: Probe candidates to find the gradient of the duals
        for i in candidates:
            probe_weights = main_weights.copy()
            probe_weights[i] = probe_weight
            duals_probe = mwu_dual_solve_weighted(A, b, probe_weights)
            dual_change = duals_probe - duals_base
            total_dual_change += dual_change

        # 5. Update main weights based on the average gradient
        avg_dual_change = total_dual_change / len(candidates)
        update_factors = np.exp(learning_rate * avg_dual_change)
        main_weights = main_weights * update_factors
        
        # Apply the simplicity bias
        # This creates a pressure to favor solutions with fewer constraints.
        main_weights = main_weights * simplicity_bias_factor
        
        # Re-normalize all weights
        main_weights = m * main_weights / np.sum(main_weights)

        # Update the influence memory for the NEXT iteration
        influence_memory = memory_decay * influence_memory + (1 - memory_decay) * np.abs(avg_dual_change)

    # 6. Final Selection: Culprits have the lowest final weights
    mean_w = np.mean(main_weights)
    std_w = np.std(main_weights)
    if std_w < 1e-9: return []

    threshold = mean_w - 1.5 * std_w
    return [i for i, w in enumerate(main_weights) if w < threshold]



# ---------------- Benchmark Suite ---------------- #

def create_infeasible_system(m: int, n: int, bad: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Creates a dense m x n system with a FAIR ground truth set that includes 'accomplice' rows."""
    true_solution_x = np.random.randn(n, 1)
    A = np.random.randn(m, n)
    b = A @ true_solution_x
    
    explicitly_bad_indices = list(range(m - bad, m))
    ground_truth_indices = set(explicitly_bad_indices)
    
    # The base rows used to construct the inconsistent rows are also part of the ground truth
    num_base_rows = min(m - bad, n) 
    base_indices_for_combos = list(range(num_base_rows))

    for i in explicitly_bad_indices:
        ground_truth_indices.update(base_indices_for_combos)
        
        coeffs = np.random.randn(1, len(base_indices_for_combos))
        new_A_row = coeffs @ A[base_indices_for_combos, :]
        consistent_b = coeffs @ b[base_indices_for_combos]
        inconsistent_b = consistent_b + (np.random.rand() + 0.5) * np.sign(np.random.randn())
        A[i, :] = new_A_row
        b[i] = inconsistent_b
        
    return A, b, sorted(list(ground_truth_indices))

def benchmark(test_cases: List[Tuple[str, np.ndarray, np.ndarray, List[int]]], solvers: List[Tuple[str, Callable]]) -> Tuple[List[dict], dict]:
    """Benchmarks solvers on test cases, using F1 score for accuracy."""
    def f1_measure(pred: List[int], truth: List[int]) -> float:
        pred_set, truth_set = set(pred), set(truth)
        if not truth_set and not pred_set: return 100.0
        tp = len(pred_set & truth_set)
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0
        recall = tp / len(truth_set) if len(truth_set) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return f1 * 100
    
    rows, avgs = [], {f"{s_name}_{m}": 0.0 for s_name, _ in solvers for m in ["time", "score"]}
    
    for name, A, b, bad_indices in test_cases:
        row = {"case": name, "m": A.shape[0], "n": A.shape[1], "truth_count": len(bad_indices)}
        for s_name, s_func in solvers:
            start = time.time(); pred = s_func(A, b); elapsed = time.time() - start
            score = f1_measure(pred if pred is not None else [], bad_indices)
            row[f"{s_name}_time"] = elapsed
            row[f"{s_name}_pred"] = len(pred or [])
            row[f"{s_name}_score"] = score
            avgs[f"{s_name}_time"] += elapsed; avgs[f"{s_name}_score"] += score
        rows.append(row)
        
    for key in avgs: avgs[key] /= len(test_cases)
    return rows, avgs

# ---------------- Main Execution Block ---------------- #

if __name__ == "__main__":
    pd.set_option('display.width', 150); pd.set_option('display.precision', 4)
    
    # Setup for the "hard" test case: overdetermined system with strong, conflicting noise
    sizes = [(1000, 150)] 
    NUM_RUNS_PER_SETTING = 2
    bads = [20, 50, 180, 310] 
    cases = []
    
    print("Generating test cases with 'fair' ground truth...")
    for m, n in sizes:
        for bad_count in bads:
            if bad_count >= m//2: continue
            for i in range(NUM_RUNS_PER_SETTING):
                A, b, idx = create_infeasible_system(m, n, bad_count)
                cases.append((f"dns_{m}x{n}_bad{bad_count}_run{i+1}", A, b, idx))

    print(f"Benchmarking {len(cases)} cases...")
    solvers = [
        ("L1S", L1_find), 
        ("L2S", L2_find),
        ("MWU", MWU_find),
        #("L1_it", L1_find_iterative),
        ("L1IF", L1_find_influence_fast),
        #("L1AW", L1_find_adaptive_weights)
        ("MWUAW", MWU_find_adaptive_weights),
        #("MWUAC", MWU_find_correlation_weights),
        ("MWUAG", MWU_find_gradient_weights),
        ("MWUAR", MWU_find_gradient_regularized)
    ]
    results, average = benchmark(cases, solvers)
    
    print("\n--- Individual Case Results ---")
    df_results = pd.DataFrame(results).set_index('case')
    print(df_results)
    
    print("\n\n--- Average Performance ---")
    df_average = pd.DataFrame(average, index=[0])
    print(df_average)