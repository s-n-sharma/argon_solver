import time
import networkx as nx
from scipy.sparse import random
import numpy as np

def analyze_system_constraints(A):
    """
    Analyzes a linear system Ax=b to find under/over-constrained subsystems
    using a graph-based method.

    Args:
        A (scipy.sparse.spmatrix): The coefficient matrix of the system.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    m, n = A.shape
    
    # 1. Create an undirected graph
    G = nx.Graph()

    # Add nodes for each variable ('v_j') and each constraint ('c_i')
    # Using prefixes 'v_' and 'c_' to distinguish node types
    variable_nodes = [f'v_{j}' for j in range(n)]
    constraint_nodes = [f'c_{i}' for i in range(m)]
    G.add_nodes_from(variable_nodes)
    G.add_nodes_from(constraint_nodes)

    # 2. Add edges for non-zero coefficients
    # A.nonzero() efficiently finds the coordinates of non-zero elements
    rows, cols = A.nonzero()
    for i, j in zip(rows, cols):
        G.add_edge(f'c_{i}', f'v_{j}')

    # 3. Find connected components (subsystems)
    # For an undirected graph, this is equivalent to finding the subsystems.
    # The slide's mention of Tarjan's is for directed graphs (SCCs); for
    # undirected graphs, a simpler BFS or DFS based approach is used.
    connected_components = list(nx.connected_components(G))

    # 4. Count constraints and variables in each component
    results = {
        'underconstrained': 0,
        'overconstrained': 0,
        'perfectly_constrained': 0,
        'details': []
    }

    for i, component in enumerate(connected_components):
        num_constraints = sum(1 for node in component if node.startswith('c_'))
        num_variables = sum(1 for node in component if node.startswith('v_'))
        
        status = "perfectly_constrained"
        if num_constraints < num_variables:
            status = "underconstrained"
            results['underconstrained'] += 1
        elif num_constraints > num_variables:
            status = "overconstrained"
            results['overconstrained'] += 1
        else:
            results['perfectly_constrained'] += 1
            
        results['details'].append({
            'subsystem_id': i,
            'status': status,
            'num_constraints': num_constraints,
            'num_variables': num_variables
        })
        
    return results

# --- Main execution ---
if __name__ == "__main__":
    # Define matrix properties
    N = 2000
    DENSITY = 0.005 # Sparsity (0.5% non-zero elements)

    # Create a large, sparse 1000x1000 matrix
    print(f"Generating a {N}x{N} sparse matrix with {DENSITY*100}% density...")
    A = random(N, N, density=DENSITY, format='coo')
    print(f"Matrix created with {A.nnz} non-zero elements.")
    print("-" * 40)

    # Time the analysis
    start_time = time.time()
    analysis_results = analyze_system_constraints(A)
    end_time = time.time()
    
    elapsed_time = end_time - start_time

    # Print the results
    print("Analysis Complete!")
    print(f"Total subsystems found: {len(analysis_results['details'])}")
    print(f"  - Underconstrained:    {analysis_results['underconstrained']}")
    print(f"  - Overconstrained:     {analysis_results['overconstrained']}")
    print(f"  - Perfectly constrained: {analysis_results['perfectly_constrained']}")
    print("-" * 40)
    print(f"🚀 Runtime: {elapsed_time:.6f} seconds")
