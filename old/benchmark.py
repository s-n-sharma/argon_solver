import time
import numpy as np
from better_collapsing import GraphBasedSolver
from qr_solver import QRSolver
from graph_qr_solver import GraphQRSolver
import matrices

def run_benchmark(solver_class, matrix_funcs):
    """Runs the benchmark for a given solver and a list of matrix functions."""
    print(f"Benchmarking Solver: {solver_class.__name__}\n")
    timings = {}

    for func_name, func in matrix_funcs.items():
        print(f"--- Testing: {func_name} ---")
        try:
            A, b = func()
            
            start_time = time.time()
            solver = solver_class(A, b)
            solution, info = solver.solve()
            end_time = time.time()
            
            duration = end_time - start_time
            timings[func_name] = duration
            
            print(f"Time taken: {duration:.6f} seconds")

            # Verify the solution and report undetermined variables
            undetermined_vars = np.where(np.isnan(solution))[0]
            if undetermined_vars.size > 0:
                print(f"Undetermined variables ({undetermined_vars.size}): {undetermined_vars.tolist()}")
            else:
                print("All variables were determined.")

            # Calculate residual error for verification
            # Replace NaN with 0 for the calculation, as Ax=b should hold for determined variables
            solution_for_residual = np.nan_to_num(solution)
            
            # Ensure matrix and vector shapes are compatible
            if A.shape[1] == solution_for_residual.shape[0]:
                residual = b - A @ solution_for_residual
                residual_norm = np.linalg.norm(residual)
                print(f"Residual norm ||Ax - b||: {residual_norm:.6e}")
            else:
                print("Could not calculate residual due to shape mismatch.")

        except Exception as e:
            print(f"An error occurred during '{func_name}': {e}")
            timings[func_name] = 'Error'
        
        print("-" * 20)

    print("\n--- Benchmark Summary ---")
    for func_name, duration in timings.items():
        if isinstance(duration, float):
            print(f"{func_name:<30}: {duration:.6f} seconds")
        else:
            print(f"{func_name:<30}: {duration}")
    print("-" * 25)

if __name__ == '__main__':
    # Define the matrix functions to benchmark
    matrix_functions = {
        "Inconsistent System": matrices.get_inconsistent_system,
        "Old Solver Bad Example": matrices.get_old_solver_bad_example,
        "Circular Constraints Example": matrices.get_circular_constraints_example,
    }

    # Run the benchmark
    print("\n" + "="*40)
    run_benchmark(GraphBasedSolver, matrix_functions)
    print("\n" + "="*40)
    run_benchmark(QRSolver, matrix_functions)
    print("\n" + "="*40)
    run_benchmark(GraphQRSolver, matrix_functions)
