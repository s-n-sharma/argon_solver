import numpy as np
import scipy as sp

class ConstraintGeneration:
    """efficiently create data
    """
    def create_circular_network( num_constraints, consistent = True):
        """This creates a circular factor graph where each variable is tied to 2 constraints in a closed loop
            num_variables = num_constraints
        """
        num_variables = num_constraints
        rows = np.repeat(np.arange(num_constraints), 2)
        
        cols_diag = np.arange(num_variables)
        cols_offdiag = np.roll(cols_diag, -1)
        cols = np.stack((cols_diag, cols_offdiag), axis=-1).flatten()
        
        data = np.tile([1, -1], num_constraints)
        
        A = sp.sparse.csc_matrix((data, (rows, cols)), shape=(num_constraints, num_variables))
        if consistent:
            x_true = np.random.rand(num_variables)
            b = A @ x_true
        else:
            x_true = np.random.rand(num_variables)
            b_consistent = A @ x_true
            y = np.ones(num_constraints)
            b = b_consistent + y 

        return A, b
    
    def create_tree_network(num_constraints):
        """This creates a factor graph which is a tree"""
        num_variables = num_constraints
        
        diagonals = [-np.ones(num_variables), np.ones(num_variables)]
        offsets = [-1, 0]
        
        A = sp.sparse.diags(diagonals, offsets, shape=(num_constraints, num_variables), format='csc')
        b = np.random.rand(num_constraints)
        return A, b
        
    def create_two_var_constraints(self, num_constraints, num_variables, consistent=True):
        """This creates a constraint network where each equation is in form [00...1...-1...]x = [b] (only 2 nonzero entries per row)"""
        rows_idx = []
        cols_idx = []
        data = []
        if consistent:
            for i in range(num_constraints):
                j1, j2 = np.random.choice(num_variables, 2, replace=False)
                rows_idx.extend([i, i])
                cols_idx.extend([j1, j2])
                data.extend([1, -1])
                
            A = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(num_constraints, num_variables))
            x_true = np.random.rand(num_variables)
            b = A @ x_true
            return A, b
        else:
            for i in range(num_constraints - 1):
                j1, j2 = np.random.choice(num_variables, 2, replace=False)
                rows_idx.extend([i, i])
                cols_idx.extend([j1, j2])
                data.extend([1, -1])
                
            A_temp = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(num_constraints, num_variables))
            row_0 = A_temp.getrow(0)
            rows_idx.extend([num_constraints - 1] * row_0.nnz)
            cols_idx.extend(row_0.indices)
            data.extend(row_0.data)
            
            A = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(num_constraints, num_variables))

            b = np.random.rand(num_constraints)
            b[num_constraints - 1] = b[0] + 1.0 

        return A, b
    
    def create_midpoint_two_var(num_constraints, num_variables):
        """This creates a constraint network where each equation is in form [00...1...-1...]x = [b]
        or [00...1...-2..000.1..000]x = [0] (only 3 non zero entries per row )
        """
        rows_idx = []
        cols_idx = []
        data = []
        b = np.random.rand(num_constraints)
        
        for i in range(num_constraints):
            if np.random.rand() > 0.5:
                # 2-var constraint (1, -1)
                j1, j2 = np.random.choice(num_variables, 2, replace=False)
                rows_idx.extend([i, i])
                cols_idx.extend([j1, j2])
                data.extend([1, -1])
            else:
                # 3-var constraint (1, -2, 1)
                j1, j2, j3 = np.random.choice(num_variables, 3, replace=False)
                rows_idx.extend([i, i, i])
                cols_idx.extend([j1, j2, j3])
                data.extend([1, -2, 1])
                b[i] = 0.0
                
        A = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(num_constraints, num_variables))
        return A, b
        
    def create_random_sparse_constraints(num_constraints, num_variables, consistent=True):
        """create random constraints sparse graphs that are sparse"""
        avg_degree = 4
        density = min(avg_degree / num_variables, 1.0)
        
        data_rvs = np.random.randn # Use a function, not the result
        
        if consistent:
            A = sp.sparse.random(num_constraints, num_variables, 
                                density=density, 
                                format='csc',
                                data_rvs=data_rvs)
            
            x_true = np.random.rand(num_variables)
            b = A @ x_true
        else:

            if num_constraints < 2:
                raise ValueError("Cannot guarantee inconsistency with < 2 constraints")
            A_m_minus_1 = sp.sparse.random(num_constraints - 1, num_variables, 
                                            density=density, 
                                            format='csc',
                                            data_rvs=data_rvs)

            A_last_row = A_m_minus_1.getrow(0)

            A = sp.sparse.vstack([A_m_minus_1, A_last_row], format='csc')

            b = np.random.rand(num_constraints)
            b[num_constraints - 1] = b[0] + 1.0

        return A, b