import numpy as np
import scipy.sparse as sp
import scipy.linalg
import networkx as nx
from qr_solver import QRSolver

class GraphQRSolver:
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
        # This part is identical to GraphBasedSolver
        self.graph = nx.Graph()
        # ... (code omitted for brevity, it's the same)
        self.constraint_nodes = range(self.m)
        self.variable_nodes = range(self.m, self.m + self.n)
        self.graph.add_nodes_from(self.constraint_nodes, bipartite=0)
        self.graph.add_nodes_from(self.variable_nodes, bipartite=1)
        edges = [(i, self.m + j) for i, j in zip(self.A.row, self.A.col)]
        self.graph.add_edges_from(edges)


    def solve(self):
        # This part is identical to GraphBasedSolver
        connected_components = list(nx.connected_components(self.graph))
        connected_components.sort(key=len, reverse=True)
        for i, cc_nodes in enumerate(connected_components):
            info = self._process_subsystem(cc_nodes, i)
            self.subsystem_info.append(info)
        return self.solution, self.subsystem_info

    def _process_subsystem(self, cc_nodes, cc_id):
        # --- 1 & 2. Graph Preprocessing (from GraphBasedSolver) ---
        constraint_indices = sorted([n for n in cc_nodes if n < self.m])
        variable_indices = sorted([n - self.m for n in cc_nodes if n >= self.m])
        info = {'id': cc_id, 'num_constraints': len(constraint_indices), 'num_variables': len(variable_indices)}

        if not constraint_indices or not variable_indices:
            info['status'] = 'trivial'
            return info
            
        A_sub = self.A.tocsr()[constraint_indices, :][:, variable_indices].tolil()
        b_sub = self.b[constraint_indices].copy()
        sub_to_orig_var = {i: v for i, v in enumerate(variable_indices)}
        sub_graph = nx.bipartite.from_biadjacency_matrix(A_sub.tocsr())
        sub_solution = np.full(len(variable_indices), np.nan)
        elimination_plan = []

        # Peeling loop (same as GraphBasedSolver, can include min-degree too)
        while True:
            if sub_graph.number_of_nodes() == 0: break
            leaves = [node for node, deg in dict(sub_graph.degree()).items() if deg == 1]
            if not leaves: break
            
            con_leaf = next((n for n in leaves if n < A_sub.shape[0]), None)
            if con_leaf is not None:
                var_node = list(sub_graph.neighbors(con_leaf))[0]
                var_sub_idx = var_node - A_sub.shape[0]
                coeff = A_sub[con_leaf, var_sub_idx]
                if not np.isclose(coeff, 0):
                    val = b_sub[con_leaf] / coeff
                    sub_solution[var_sub_idx] = val
                    for neighbor_con in list(sub_graph.neighbors(var_node)):
                        b_sub[neighbor_con] -= A_sub[neighbor_con, var_sub_idx] * val
                sub_graph.remove_node(var_node)
                sub_graph.remove_node(con_leaf)
            else: # Variable leaf
                var_leaf = leaves[0]
                con_node = list(sub_graph.neighbors(var_leaf))[0]
                var_sub_idx = var_leaf - A_sub.shape[0]
                elimination_plan.append((var_sub_idx, con_node))
                sub_graph.remove_node(var_leaf)
                sub_graph.remove_node(con_node)

        # --- 3. Solve Core System (with QRSolver logic) ---
        core_cons_idx = sorted([n for n in sub_graph.nodes if n < A_sub.shape[0]])
        core_vars_idx = sorted([n - A_sub.shape[0] for n in sub_graph.nodes if n >= A_sub.shape[0]])
        
        core_solved = False
        if core_cons_idx and core_vars_idx:
            A_core = A_sub[core_cons_idx, :][:, core_vars_idx].toarray()
            b_core = b_sub[core_cons_idx]
            
            # Use the robust QR method on the core
            core_solver = QRSolver(A_core, b_core) 
            core_sol, core_info = core_solver.solve()
            
            info.update(core_info) # Merge info from the core solver
            
            if core_info['status'] != 'inconsistent':
                # Map core solution back to the subsystem solution
                for i, sub_idx in enumerate(core_vars_idx):
                    sub_solution[sub_idx] = core_sol[i]
                core_solved = True
        else: # No core left to solve
            info['status'] = 'solved_by_peeling'
            core_solved = True
        
        # --- 4. Back-substitute (from GraphBasedSolver) ---
        if core_solved:
            for var_sub_idx, con_sub_idx in reversed(elimination_plan):
                row_vals = A_sub[con_sub_idx, :].toarray().flatten()
                coeff = row_vals[var_sub_idx]
                other_terms = np.nansum([val * sub_solution[j] for j, val in enumerate(row_vals) if j != var_sub_idx])
                if not np.isclose(coeff, 0):
                    sub_solution[var_sub_idx] = (b_sub[con_sub_idx] - other_terms) / coeff
        
        # --- 5. Finalize Solution ---
        for sub_idx, val in enumerate(sub_solution):
            self.solution[sub_to_orig_var[sub_idx]] = val
        
        return info