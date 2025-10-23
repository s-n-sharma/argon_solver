from __future__ import annotations
from typing import Iterable

from typing import Iterable, Optional

import numpy as np
import scipy as sp

class ConstraintGeneration:
    """efficiently create data
    """

    @staticmethod
    def _prepare_rng(random_state: Optional[int | np.random.Generator]) -> np.random.Generator:
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)

    @staticmethod
    def _split_counts(total: int, groups: int) -> list[int]:
        if groups <= 0:
            raise ValueError("groups must be positive")
        if groups > total:
            raise ValueError("Cannot split into more groups than total constraints")
        base = total // groups
        remainder = total % groups
        counts = [base + (1 if i < remainder else 0) for i in range(groups)]
        return counts

    @staticmethod
    def _report_inconsistent_rows(
        A: sp.sparse.spmatrix,
        b: np.ndarray,
        rows: Iterable[int],
        reason: str,
        max_rows: int = 6,
    ) -> None:
        """Log which constraints were perturbed to induce inconsistency."""

        unique_rows = sorted({int(r) for r in rows})
        if not unique_rows:
            return

        display_rows = unique_rows[:max_rows]
        detail_lines = []
        for r in display_rows:
            row = A.getrow(r)
            entries = ", ".join(
                f"col {c}: {v:.3g}" for c, v in zip(row.indices, row.data)
            )
            detail_lines.append(f"row {r} | b={float(b[r]):.6g} | {entries}")

        suffix = ""
        if len(unique_rows) > max_rows:
            suffix = f"\n    ... ({len(unique_rows) - max_rows} more rows perturbed)"

        detail = "\n    " + "\n    ".join(detail_lines) + suffix
        print(
            f"[matrix_generation] Inconsistent constraints due to {reason}:{detail}"
        )
    @staticmethod
    def create_circular_network(
        num_constraints: int,
        consistent: bool = True,
        num_iis: int = 1,
        delta: float = 1.0,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> tuple[sp.sparse.csc_matrix, np.ndarray, tuple[int, ...]]:
        """Create one or more circular constraint components.

        Parameters
        ----------
        num_constraints : int
            Total number of constraints (and variables) across all components.
        consistent : bool, optional
            When ``True`` returns a consistent system. When ``False``
            ``num_iis`` independent inconsistent components are created.
        num_iis : int, optional
            Number of independent inconsistent subsystems to embed when
            ``consistent`` is ``False``. Ignored otherwise.
        delta : float, optional
            Magnitude of the RHS perturbation that triggers infeasibility.
        random_state : int or Generator, optional
            Source of randomness for reproducibility.
        """

        rng = ConstraintGeneration._prepare_rng(random_state)

        if consistent:
            num_variables = num_constraints
            rows = np.repeat(np.arange(num_constraints), 2)
            cols_diag = np.arange(num_variables)
            cols_offdiag = np.roll(cols_diag, -1)
            cols = np.stack((cols_diag, cols_offdiag), axis=-1).flatten()
            data = np.tile([1, -1], num_constraints)
            A = sp.sparse.csc_matrix((data, (rows, cols)), shape=(num_constraints, num_variables))
            x_true = rng.random(num_variables)
            b = np.asarray(A @ x_true, dtype=float).reshape(-1)
            return A, b, ()

        if num_iis < 1:
            raise ValueError("num_iis must be at least 1 when requesting inconsistencies")

        component_sizes = ConstraintGeneration._split_counts(num_constraints, num_iis)

        blocks = []
        rhs_parts = []
        inconsistent_rows = []
        row_offset = 0

        for block_idx, size in enumerate(component_sizes):
            rows = np.repeat(np.arange(size), 2)
            cols_diag = np.arange(size)
            cols_offdiag = np.roll(cols_diag, -1)
            cols = np.stack((cols_diag, cols_offdiag), axis=-1).flatten()
            data = np.tile([1, -1], size)
            block = sp.sparse.csc_matrix((data, (rows, cols)), shape=(size, size))

            x_true = rng.random(size)
            b_block = np.asarray(block @ x_true, dtype=float).reshape(-1)

            # Perturb a single constraint to break feasibility within this component.
            local_row = int(rng.integers(0, size))
            b_block[local_row] += delta
            inconsistent_rows.append(row_offset + local_row)

            blocks.append(block)
            rhs_parts.append(b_block)
            row_offset += size

        A = sp.sparse.block_diag(blocks, format="csc")
        b = np.concatenate(rhs_parts)

        ConstraintGeneration._report_inconsistent_rows(
            A,
            b,
            inconsistent_rows,
            reason=f"{num_iis} circular loop(s) with shifted RHS",
        )

        return A, b, tuple(inconsistent_rows)
    
    @staticmethod
    def create_tree_network(num_constraints: int) -> tuple[sp.sparse.csc_matrix, np.ndarray, tuple[int, ...]]:
        """This creates a factor graph which is a tree"""
        num_variables = num_constraints
        
        diagonals = [-np.ones(num_variables), np.ones(num_variables)]
        offsets = [-1, 0]
        
        A = sp.sparse.diags(diagonals, offsets, shape=(num_constraints, num_variables), format='csc')
        b = np.random.rand(num_constraints)
        return A, b, ()
        
    @staticmethod
    def create_two_var_constraints(
        num_constraints: int,
        num_variables: int,
        consistent: bool = True,
        num_iis: int = 1,
        delta: float = 1.0,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> tuple[sp.sparse.csc_matrix, np.ndarray, tuple[int, ...]]:
        """This creates a constraint network where each equation is in form [00...1...-1...]x = [b] (only 2 nonzero entries per row)"""
        rng = ConstraintGeneration._prepare_rng(random_state)
        rows_idx = []
        cols_idx = []
        data = []
        if consistent:
            for i in range(num_constraints):
                j1, j2 = rng.choice(num_variables, 2, replace=False)
                rows_idx.extend([i, i])
                cols_idx.extend([j1, j2])
                data.extend([1, -1])
                
            A = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(num_constraints, num_variables))
            x_true = rng.random(num_variables)
            b = np.asarray(A @ x_true, dtype=float).reshape(-1)
            return A, b, ()
        else:
            if num_iis < 1:
                raise ValueError("num_iis must be at least 1 when requesting inconsistencies")
            if num_iis >= num_constraints:
                raise ValueError("num_iis must be less than num_constraints for this generator")

            base_rows = num_constraints - num_iis
            for i in range(base_rows):
                j1, j2 = rng.choice(num_variables, 2, replace=False)
                rows_idx.extend([i, i])
                cols_idx.extend([j1, j2])
                data.extend([1, -1])
            A_base = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(base_rows, num_variables))
            A = A_base
            b_base = rng.random(base_rows)

            duplicated_rows = []
            duplicated_rhs = []
            inconsistent_rows = []

            for dup_idx in range(num_iis):
                source_row_id = int(rng.integers(0, base_rows))
                duplicated_rows.append(A_base.getrow(source_row_id))
                duplicated_rhs.append(b_base[source_row_id] + delta)
                inconsistent_rows.append(base_rows + dup_idx)

            if duplicated_rows:
                A_duplicates = sp.sparse.vstack(duplicated_rows, format="csc")
                A = sp.sparse.vstack([A_base, A_duplicates], format="csc")
            b = np.concatenate([b_base, np.asarray(duplicated_rhs, dtype=float)])

            ConstraintGeneration._report_inconsistent_rows(
                A,
                b,
                inconsistent_rows,
                reason=f"{num_iis} duplicated 2-var constraints with shifted RHS",
            )

            return A, b, tuple(inconsistent_rows)
    
    @staticmethod
    def create_midpoint_two_var(
        num_constraints: int,
        num_variables: int,
        num_iis: int = 0,
        delta: float = 1.0,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> tuple[sp.sparse.csc_matrix, np.ndarray, tuple[int, ...]]:
        """This creates a constraint network where each equation is in form [00...1...-1...]x = [b]
        or [00...1...-2..000.1..000]x = [0] (only 3 non zero entries per row )
        """
        rng = ConstraintGeneration._prepare_rng(random_state)
        if num_iis < 0:
            raise ValueError("num_iis must be non-negative")
        if num_iis >= num_constraints:
            raise ValueError("num_iis must be less than the number of constraints")

        base_rows = num_constraints - num_iis
        rows_idx: list[int] = []
        cols_idx: list[int] = []
        data: list[float] = []
        b = rng.random(base_rows)
        source_rows: list[int] = []

        for i in range(base_rows):
            if rng.random() > 0.5:
                # 2-var constraint (1, -1)
                j1, j2 = rng.choice(num_variables, 2, replace=False)
                rows_idx.extend([i, i])
                cols_idx.extend([j1, j2])
                data.extend([1, -1])
                source_rows.append(i)
            else:
                # 3-var constraint (1, -2, 1)
                j1, j2, j3 = rng.choice(num_variables, 3, replace=False)
                rows_idx.extend([i, i, i])
                cols_idx.extend([j1, j2, j3])
                data.extend([1, -2, 1])
                b[i] = 0.0
                source_rows.append(i)
                
        A_base = sp.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(base_rows, num_variables))
        if num_iis == 0:
            return A_base, b, ()

        inconsistent_rows = []
        appended_rows = []
        appended_b = []

        for dup_idx in range(num_iis):
            source_row_id = int(rng.choice(source_rows))
            appended_rows.append(A_base.getrow(source_row_id))
            appended_b.append(float(b[source_row_id]) + delta)
            inconsistent_rows.append(base_rows + dup_idx)

        A_duplicates = sp.sparse.vstack(appended_rows, format="csc")
        A = sp.sparse.vstack([A_base, A_duplicates], format="csc")
        b_full = np.concatenate([b, np.asarray(appended_b, dtype=float)])

        ConstraintGeneration._report_inconsistent_rows(
            A,
            b_full,
            inconsistent_rows,
            reason=f"{num_iis} duplicated mixed constraints with shifted RHS",
        )

        return A, b_full, tuple(inconsistent_rows)
        
    @staticmethod
    def create_random_sparse_constraints(
        num_constraints: int,
        num_variables: int,
        consistent: bool = True,
        num_iis: int = 1,
        delta: float = 1.0,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> tuple[sp.sparse.csc_matrix, np.ndarray, tuple[int, ...]]:
        """create random constraints sparse graphs that are sparse"""
        avg_degree = 4
        density = min(avg_degree / num_variables, 1.0)

        rng = ConstraintGeneration._prepare_rng(random_state)

        def data_rvs(size: int) -> np.ndarray:
            return rng.standard_normal(size)

        if consistent:
            A = sp.sparse.random(num_constraints, num_variables, 
                                density=density, 
                                format='csc',
                                data_rvs=data_rvs)
            
            x_true = rng.random(num_variables)
            b = np.asarray(A @ x_true, dtype=float).reshape(-1)
            return A, b, ()
        else:

            if num_constraints < 2:
                raise ValueError("Cannot guarantee inconsistency with < 2 constraints")
            if num_iis < 1:
                raise ValueError("num_iis must be at least 1 when requesting inconsistencies")
            if num_iis >= num_constraints:
                raise ValueError("num_iis must be less than the number of constraints")

            base_rows = num_constraints - num_iis
            A_base = sp.sparse.random(
                base_rows,
                num_variables,
                density=density,
                format='csc',
                data_rvs=data_rvs,
            )

            b_base = rng.random(base_rows)

            duplicated_rows = []
            duplicated_rhs = []
            inconsistent_rows = []

            for dup_idx in range(num_iis):
                source_row_id = int(rng.integers(0, base_rows))
                duplicated_rows.append(A_base.getrow(source_row_id))
                duplicated_rhs.append(b_base[source_row_id] + delta)
                inconsistent_rows.append(base_rows + dup_idx)

            if duplicated_rows:
                A_duplicates = sp.sparse.vstack(duplicated_rows, format='csc')
                A = sp.sparse.vstack([A_base, A_duplicates], format='csc')
            else:
                A = A_base

            b = np.concatenate([b_base, np.asarray(duplicated_rhs, dtype=float)])

            ConstraintGeneration._report_inconsistent_rows(
                A,
                b,
                inconsistent_rows,
                reason=f"{num_iis} duplicated sparse constraints with shifted RHS",
            )

            return A, b, tuple(inconsistent_rows)