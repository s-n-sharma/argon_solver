import numpy as np
import scipy.sparse as sp
import networkx as nx
import json

class GraphBasedSolver:
    """
    Solves a system of linear equations Ax=b using a graph-based method.

    The method involves decomposing the system into independent subsystems based
    on the sparsity pattern of matrix A, preconditioning each subsystem by
    "peeling" leaf nodes, solving the remaining core system, and then
    back-substituting to find the complete solution.
    """
    def __init__(self, A, b):
        """
        Initializes the solver.

        Args:
            A (scipy.sparse matrix or np.ndarray): The m x n coefficient matrix.
            b (np.ndarray): The m x 1 constant vector.
        """
        if not sp.issparse(A):
            A = sp.csc_matrix(A)
        # Use COO format for easy access to row/col pairs for graph building
        self.A = A.tocoo()
        self.b = b
        self.m, self.n = A.shape
        self.solution = np.full(self.n, np.nan)
        self.subsystem_info = []

        self._build_graph()

    def _build_graph(self):
        """Builds a bipartite graph from the matrix A's sparsity pattern."""
        self.graph = nx.Graph()
        # Constraint nodes are labeled 0 to m-1
        # Variable nodes are labeled m to m+n-1 to avoid collision
        self.constraint_nodes = range(self.m)
        self.variable_nodes = range(self.m, self.m + self.n)
        self.graph.add_nodes_from(self.constraint_nodes, bipartite=0)
        self.graph.add_nodes_from(self.variable_nodes, bipartite=1)

        # An edge exists if A[i,j] is non-zero
        edges = [(i, self.m + j) for i, j in zip(self.A.row, self.A.col)]
        self.graph.add_edges_from(edges)

    def solve(self):
        """
        Decomposes the system into subsystems and solves them.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The solution vector x.
                - list: A list of dictionaries with info about each subsystem.
        """
        # Find connected components, which represent independent subsystems
        connected_components = list(nx.connected_components(self.graph))
        connected_components.sort(key=len, reverse=True)

        for i, cc_nodes in enumerate(connected_components):
            info = self._process_subsystem(cc_nodes, i)
            self.subsystem_info.append(info)

        return self.solution, self.subsystem_info

    def _process_subsystem(self, cc_nodes, cc_id):
        """Analyzes, preconditions, and solves a single subsystem."""
        # 1. Identify subproblem from the connected component
        # Original indices from matrix A
        constraint_indices = sorted([n for n in cc_nodes if n < self.m])
        variable_indices = sorted([n - self.m for n in cc_nodes if n >= self.m])

        info = {
            'id': cc_id,
            'num_constraints': len(constraint_indices),
            'num_variables': len(variable_indices),
            'status': 'unprocessed'
        }

        if not constraint_indices or not variable_indices:
            info['status'] = 'trivial'
            return info

        # Extract sub-matrix and sub-vector. Use LIL format for efficient modification.
        A_sub = self.A.tocsr()[constraint_indices, :][:, variable_indices].tolil()
        b_sub = self.b[constraint_indices].copy()
        
        # Local mapping: subsystem_index -> original_variable_index
        sub_to_orig_var = {i: v for i, v in enumerate(variable_indices)}
        
        # 2. Precondition via iterative peeling
        sub_graph = nx.bipartite.from_biadjacency_matrix(A_sub.tocsr())
        sub_solution = np.full(len(variable_indices), np.nan)
        elimination_plan = [] # For back-substitution of variable leaves

        while True:
            if sub_graph.number_of_nodes() == 0:
                break
            
            degrees = dict(sub_graph.degree())
            leaves = [node for node, deg in degrees.items() if deg == 1]
            
            if not leaves:
                break # No more leaves, the rest is the core

            # Prioritize constraint leaves (Type 1 peel)
            # These are nodes from the first partition of the bipartite graph
            constraint_leaves = [n for n in leaves if n < A_sub.shape[0]]
            
            if constraint_leaves:
                con_sub_idx = constraint_leaves[0]
                var_node_local = list(sub_graph.neighbors(con_sub_idx))[0]
                var_sub_idx = var_node_local - A_sub.shape[0]

                # Solve: A[c,v]*x[v] = b[c]
                val = b_sub[con_sub_idx] / A_sub[con_sub_idx, var_sub_idx]
                sub_solution[var_sub_idx] = val

                # Substitute this solved value into the rest of the system
                for neighbor_con in list(sub_graph.neighbors(var_node_local)):
                    if neighbor_con != con_sub_idx:
                        b_sub[neighbor_con] -= A_sub[neighbor_con, var_sub_idx] * val
                
                # The variable is solved, so remove its node from the graph
                sub_graph.remove_node(var_node_local)
            else:
                # No constraint leaves, process variable leaves (Type 2 peel)
                var_node_local = leaves[0]
                con_sub_idx = list(sub_graph.neighbors(var_node_local))[0]
                var_sub_idx = var_node_local - A_sub.shape[0]

                # Plan to solve this variable later using its one equation
                elimination_plan.append((var_sub_idx, con_sub_idx))
                sub_graph.remove_node(var_node_local)
                sub_graph.remove_node(con_sub_idx)

        # 3. Solve the core system
        core_cons_sub_idx = sorted([n for n, attr in sub_graph.nodes(data=True) if attr['bipartite'] == 0])
        core_vars_sub_idx = sorted([n - A_sub.shape[0] for n, attr in sub_graph.nodes(data=True) if attr['bipartite'] == 1])

        info.update({
            'peeled_vars': len(variable_indices) - len(core_vars_sub_idx),
            'core_constraints': len(core_cons_sub_idx),
            'core_variables': len(core_vars_sub_idx)
        })

        if core_cons_sub_idx:
            A_core = A_sub[core_cons_sub_idx, :][:, core_vars_sub_idx].toarray()
            b_core = b_sub[core_cons_sub_idx]
            m_core, n_core = A_core.shape
            
            try:
                if m_core == n_core:
                    info['classification'] = 'properly_constrained_core'
                    core_sol = np.linalg.solve(A_core, b_core)
                else: # Under- or over-constrained
                    info['classification'] = 'under/over-constrained_core'
                    core_sol = np.linalg.lstsq(A_core, b_core, rcond=None)[0]
                
                # Place core solution into the local solution vector
                for i, var_idx in enumerate(core_vars_sub_idx):
                    sub_solution[var_idx] = core_sol[i]
                info['status'] = 'solved'
            except np.linalg.LinAlgError:
                info['status'] = 'failed_singular_core'
        else:
            # System was fully solved by peeling
            info['status'] = 'solved'

        # 4. Back-substitute for Type 2 peeled variables
        if info['status'] == 'solved':
            for var_sub_idx, con_sub_idx in reversed(elimination_plan):
                row_vals = A_sub[con_sub_idx, :].toarray().flatten()
                coeff = row_vals[var_sub_idx]
                
                # Sum of other_vars * their_coeffs
                other_terms = np.nansum([
                    val * sub_solution[j]
                    for j, val in enumerate(row_vals) if j != var_sub_idx
                ])
                
                if coeff != 0:
                    sub_solution[var_sub_idx] = (b_sub[con_sub_idx] - other_terms) / coeff

        # 5. Place local solution into the global solution vector
        if info['status'] == 'solved':
            for sub_idx, val in enumerate(sub_solution):
                orig_idx = sub_to_orig_var[sub_idx]
                self.solution[orig_idx] = val
        
        return info


# --- Demonstration ---
if __name__ == '__main__':
    # Example 1: A block diagonal system with two independent subsystems
    # Subsystem 1 (Properly constrained): x0, x1
    #   2*x0 + 3*x1 = 8
    #   1*x0 + 1*x1 = 3
    #   Solution: x0=1, x1=2
    #
    # Subsystem 2 (Overconstrained but consistent): x2, x3
    #   4*x2 + 1*x3 = 10
    #  -1*x2 + 2*x3 = 3
    #   3*x2 + 3*x3 = 13
    #   Solution: x2=2, x3=2
    print("--- Example 1: Two Independent Subsystems ---")
    A1 = sp.csr_matrix([
        [2, 3, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 4, 1],
        [0, 0, -1, 2],
        [0, 0, 3, 3]
    ])
    b1 = np.array([8, 3, 10, 3, 13])
    
    solver1 = GraphBasedSolver(A1, b1)
    solution1, info1 = solver1.solve()
    
    print(f"Solution: {np.round(solution1, 4)}")
    print("Subsystem Info:")
    print(json.dumps(info1, indent=2))
    print("-" * 50)

    # Example 2: A system requiring the peeling algorithm
    # c0: x0 + 2*x1       = 5    (core)
    # c1:      3*x1 + x2 = 7    (core)
    # c2:           4*x2 = 8    (determines x2, constraint leaf)
    # c3: x0 +          x3 = 10 (x3 is a variable leaf)
    # c4:                  5*x4 = 15 (separate 1x1 subsystem)
    # Expected solution: x = [1.667, 1.667, 2, 8.333, 3]
    print("--- Example 2: System Requiring Peeling ---")
    A2 = sp.csr_matrix([
        [1, 2, 0, 0, 0],
        [0, 3, 1, 0, 0],
        [0, 0, 4, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 5],
    ])
    b2 = np.array([5, 7, 8, 10, 15])
    
    solver2 = GraphBasedSolver(A2, b2)
    solution2, info2 = solver2.solve()
    print(f"Solution: {np.round(solution2, 4)}")
    print("Subsystem Info:")
    print(json.dumps(info2, indent=2))
    print("-" * 50)

    #Example 3: A HUGE 1000x1500 sparse system
    print("--- Example 3: Large Sparse System ---")
    np.random.seed(42)
    size = 1500
    density = 0.005
    A3 = sp.random(size, int(1.5*size), density=density, format='csr', data_rvs=np.random.randn)
    b3 = np.random.randn(size)
    import time
    start_time = time.time()
    solver3 = GraphBasedSolver(A3, b3)
    solution3, info3 = solver3.solve()
    end_time = time.time()
    print(f"Solved large system in {end_time - start_time:.2f} seconds")

