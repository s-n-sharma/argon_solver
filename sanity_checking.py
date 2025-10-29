import numpy as np
import scipy.sparse as sp
import scipy.linalg
import networkx as nx
import json
import time  # Import the time module


class GraphBasedSolver:
    """
    Solves a system of linear equations Ax=b using a graph-based method.

    This solver uses a structural approach to simplify and solve linear systems:
    1.  It builds a bipartite graph to represent the system's structure.
    2.  It identifies independent subsystems (connected components).
    3.  It applies a "peeling" preconditioner to solve for easy variables.
    4.  It performs relational substitution to eliminate variables, but only if
        this process does not increase the density (fill-in) of the graph.
    5.  It solves the final, irreducible "core" of the system using a standard
        numerical solver (numpy.linalg.lstsq).
    6.  It diagnoses inconsistent or underconstrained systems.
    7.  It back-substitutes using the original equations to find the complete solution.
    """

    def __init__(self, A, b):
        if not sp.issparse(A):
            A = sp.csc_matrix(A)
        self.A = A.tocoo()
        self.b = b
        self.m, self.n = A.shape
        self.solution = np.full(self.n, np.nan)
        self.subsystem_info = []
        self._build_graph()

    def _build_graph(self):
        self.graph = nx.Graph()
        self.constraint_nodes = range(self.m)
        self.variable_nodes = range(self.m, self.m + self.n)
        self.graph.add_nodes_from(self.constraint_nodes, bipartite=0)
        self.graph.add_nodes_from(self.variable_nodes, bipartite=1)
        edges = [(i, self.m + j) for i, j in zip(self.A.row, self.A.col)]
        self.graph.add_edges_from(edges)

    def solve(self):
        connected_components = list(nx.connected_components(self.graph))
        connected_components.sort(key=len, reverse=True)
        for i, cc_nodes in enumerate(connected_components):
            info = self._process_subsystem(cc_nodes, i)
            self.subsystem_info.append(info)
        return self.solution, self.subsystem_info

    @staticmethod
    def solve_with_scipy(A, b):
        """Solves the system Ax=b using SciPy's direct sparse solver as a benchmark."""
        # spsolve is most efficient with CSR or CSC formats
        A_csr = A.tocsr()

        try:
            # use_umfpack=True can be faster but requires the umfpack library
            solution = sp.linalg.spsolve(A_csr, b)
            return solution
        except Exception as e:
            print(f"SciPy solver failed: {e}")
            return np.full(A.shape[1], np.nan)

    def _process_subsystem(self, cc_nodes, cc_id):
        # 1. Identification and Setup
        constraint_indices = sorted([n for n in cc_nodes if n < self.m])
        variable_indices = sorted([n - self.m for n in cc_nodes if n >= self.m])
        info = {
            "id": cc_id,
            "num_constraints": len(constraint_indices),
            "num_variables": len(variable_indices),
            "status": "unprocessed",
        }
        if not constraint_indices or not variable_indices:
            info["status"] = "trivial"
            return info

        A_sub = self.A.tocsr()[constraint_indices, :][:, variable_indices].tolil()
        b_sub = self.b[constraint_indices].copy()
        A_sub_orig = A_sub.copy()
        b_sub_orig = b_sub.copy()
        sub_to_orig_var = {i: v for i, v in enumerate(variable_indices)}

        active_vars = list(range(A_sub.shape[1]))
        active_cons = list(range(A_sub.shape[0]))
        sub_solution = np.full(len(variable_indices), np.nan)
        elimination_plan = []

        # 2. Preconditioning Stage 1: Peeling of Degree-1 Nodes
        made_progress = True
        while made_progress:
            made_progress = False

            var_degrees = {v_idx: 0 for v_idx in active_vars}
            active_vars_set = set(active_vars)
            for c_idx in active_cons:
                for v_idx in A_sub.getrow(c_idx).nonzero()[1]:
                    if v_idx in active_vars_set:
                        var_degrees[v_idx] += 1

            leaf_vars = [v for v, d in var_degrees.items() if d == 1]
            if not leaf_vars:
                break

            pivot_var_idx = leaf_vars[0]
            pivot_con_idx = -1
            for c_idx in active_cons:
                if not np.isclose(A_sub[c_idx, pivot_var_idx], 0):
                    pivot_con_idx = c_idx
                    break

            elimination_plan.append((pivot_var_idx, pivot_con_idx))
            active_vars.remove(pivot_var_idx)
            active_cons.remove(pivot_con_idx)
            made_progress = True

        # 3. Preconditioning Stage 2: Relational Substitution with "No Fill-in" Heuristic
        made_progress = True
        while made_progress:
            made_progress = False

            var_degrees = {v_idx: 0 for v_idx in active_vars}
            active_vars_set = set(active_vars)
            for c_idx in active_cons:
                for v_idx in A_sub.getrow(c_idx).nonzero()[1]:
                    if v_idx in active_vars_set:
                        var_degrees[v_idx] += 1

            pivot_candidate = -1
            for var_idx, degree in var_degrees.items():
                if degree == 2:
                    pivot_candidate = var_idx
                    break

            if pivot_candidate == -1:
                break

            connected_cons = [
                c for c in active_cons if not np.isclose(A_sub[c, pivot_candidate], 0)
            ]
            pivot_eq, subst_eq = connected_cons[0], connected_cons[1]

            vars_in_pivot_eq = set(A_sub.getrow(pivot_eq).nonzero()[1])
            vars_in_subst_eq = set(A_sub.getrow(subst_eq).nonzero()[1])

            merged_vars = (vars_in_pivot_eq | vars_in_subst_eq) - {pivot_candidate}

            nnz_change = len(merged_vars) - len(vars_in_subst_eq)

            if nnz_change > 0:
                break

            pivot_coeff = A_sub[pivot_eq, pivot_candidate]
            subst_coeff = A_sub[subst_eq, pivot_candidate]

            if np.isclose(pivot_coeff, 0):
                continue

            scale = subst_coeff / pivot_coeff
            A_sub[subst_eq, :] -= scale * A_sub[pivot_eq, :]
            b_sub[subst_eq] -= scale * b_sub[pivot_eq]

            elimination_plan.append((pivot_candidate, pivot_eq))
            active_vars.remove(pivot_candidate)
            active_cons.remove(pivot_eq)
            made_progress = True

        # 4. Solve the final irreducible core
        core_vars_sub_idx = active_vars
        core_cons_sub_idx = active_cons
        info.update(
            {
                "peeled_vars": len(variable_indices) - len(core_vars_sub_idx),
                "core_constraints": len(core_cons_sub_idx),
                "core_variables": len(core_vars_sub_idx),
            }
        )

        core_solved = False
        if not core_cons_sub_idx and not core_vars_sub_idx:
            info["status"] = "solved"
            core_solved = True
        elif core_cons_sub_idx and core_vars_sub_idx:
            A_core = A_sub[core_cons_sub_idx, :][:, core_vars_sub_idx].toarray()
            b_core = b_sub[core_cons_sub_idx]

            try:
                core_sol, _, _, _ = np.linalg.lstsq(A_core, b_core, rcond=None)
                info["status"] = "solved"
                for i, var_idx in enumerate(core_vars_sub_idx):
                    sub_solution[var_idx] = core_sol[i]
                core_solved = True
            except np.linalg.LinAlgError:
                info["status"] = "failed_core_singular"
        else:
            info["status"] = "failed_structural_problem"

        # 5. Back-substitute for ALL eliminated variables
        if core_solved:
            for var_idx, con_idx in reversed(elimination_plan):
                row = A_sub_orig[con_idx, :].toarray().flatten()
                rhs = b_sub_orig[con_idx]

                known_terms = np.nansum(
                    [
                        v * sub_solution[j]
                        for j, v in enumerate(row)
                        if j != var_idx and not np.isnan(sub_solution[j])
                    ]
                )

                coeff = row[var_idx]
                if not np.isclose(coeff, 0):
                    sub_solution[var_idx] = (rhs - known_terms) / coeff
                else:
                    sub_solution[var_idx] = np.nan
                    info["status"] = "failed_backsubstitution"
                    break

        # 6. Place local solution into the global solution vector
        if np.any(~np.isnan(sub_solution)):
            for sub_idx, val in enumerate(sub_solution):
                self.solution[sub_to_orig_var[sub_idx]] = val
        return info


# --- Verification with the test case ---
if __name__ == "__main__":
    A = []
    size = 200
    a_row = [0] * size
    a_row[0], a_row[1], a_row[2] = 1, -2, 1
    for i in range(size - 2):
        A.append(np.array(a_row))
        a_row = [0] + a_row[:-1]

    b = [0] * (198 - 1) + [100]
    b = np.array(b)
    A = scipy.sparse.csr_matrix(np.array(A))

    # --- Run the SciPy benchmark first ---
    print("--- Running SciPy Benchmark Solver ---")
    start_time_scipy = time.perf_counter()
    scipy_solution = GraphBasedSolver.solve_with_scipy(A, b)
    end_time_scipy = time.perf_counter()

    print(f"⏱️ SciPy Time: {(end_time_scipy - start_time_scipy) * 1000:.4f} ms")
    print("\nSciPy Benchmark Solution:")
    print(np.round(scipy_solution, 4))
    print("-" * 40)

    # --- Run your graph-based solver ---
    print("--- Running GraphBasedSolver ---")
    start_time_graph = time.perf_counter()
    solver = GraphBasedSolver(A, b)
    solution, info = solver.solve()
    end_time_graph = time.perf_counter()

    print(
        f"⏱️ GraphBasedSolver Time: {(end_time_graph - start_time_graph) * 1000:.4f} ms"
    )

    print("\nGraph Solver Status:")
    print(json.dumps(info, indent=2))

    print("\nGraph Solver Calculated Solution:")
    print(np.round(solution, 4))

    print("\nExpected Solution:")
    print(np.array([0.0, -3.75, -7.5, -11.25, -15.0]))
