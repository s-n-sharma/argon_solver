"""Benchmark conflict detection for CAD-style linear constraint systems."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

import numpy as np
import scipy.sparse as spa
from numpy.linalg import norm

_HERE = Path(__file__).resolve()
_CONSTRAINT_ROOT = _HERE.parent
_REPO_ROOT = _CONSTRAINT_ROOT.parent

import sys

for _path in (_CONSTRAINT_ROOT, _REPO_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:  # Allow running as both a module and a script.
    from .solver_utils import Solver
    from .sketch import (
        AnalysisConfig,
        SketchSolverOptions,
        analyze_linear_system,
    )
    from ..data import ConstraintGeneration
except ImportError:  # pragma: no cover - fallback for direct execution.
    from solver_utils import Solver
    from sketch import AnalysisConfig, SketchSolverOptions, analyze_linear_system
    from data import ConstraintGeneration


@dataclass
class ProblemSpec:
    name: str
    builder: Callable[
        [np.random.Generator],
        Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]],
    ]
    top_k: int = 5
    seed: int = 0
    sketch_size: Optional[int] = None


def _circular_builder(
    num_constraints: int,
    num_iis: int,
    delta: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]:
        if num_iis <= 0:
            A, b, inconsistent = ConstraintGeneration.create_circular_network(
                num_constraints=num_constraints,
                consistent=True,
                random_state=rng,
            )
        else:
            A, b, inconsistent = ConstraintGeneration.create_circular_network(
                num_constraints=num_constraints,
                consistent=False,
                num_iis=num_iis,
                delta=delta,
                random_state=rng,
            )
        truth = np.array(inconsistent, dtype=int) if inconsistent else None
        return A, np.asarray(b, dtype=float), truth

    return builder


def _two_var_builder(
    num_constraints: int,
    num_variables: int,
    num_iis: int,
    delta: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]:
        consistent = num_iis <= 0
        if consistent:
            A, b, inconsistent = ConstraintGeneration.create_two_var_constraints(
                num_constraints=num_constraints,
                num_variables=num_variables,
                consistent=True,
                random_state=rng,
            )
        else:
            A, b, inconsistent = ConstraintGeneration.create_two_var_constraints(
                num_constraints=num_constraints,
                num_variables=num_variables,
                consistent=False,
                num_iis=num_iis,
                delta=delta,
                random_state=rng,
            )
        truth = np.array(inconsistent, dtype=int) if inconsistent else None
        return A, np.asarray(b, dtype=float), truth

    return builder


def _midpoint_builder(
    num_constraints: int,
    num_variables: int,
    num_iis: int,
    delta: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]:
        A, b, inconsistent = ConstraintGeneration.create_midpoint_two_var(
            num_constraints=num_constraints,
            num_variables=num_variables,
            num_iis=num_iis,
            delta=delta,
            random_state=rng,
        )
        truth = np.array(inconsistent, dtype=int) if inconsistent else None
        return A, np.asarray(b, dtype=float), truth

    return builder


def _random_sparse_builder(
    num_constraints: int,
    num_variables: int,
    num_iis: int,
    delta: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, Optional[np.ndarray]]:
        consistent = num_iis <= 0
        if consistent:
            A, b, inconsistent = ConstraintGeneration.create_random_sparse_constraints(
                num_constraints=num_constraints,
                num_variables=num_variables,
                consistent=True,
                random_state=rng,
            )
        else:
            A, b, inconsistent = ConstraintGeneration.create_random_sparse_constraints(
                num_constraints=num_constraints,
                num_variables=num_variables,
                consistent=False,
                num_iis=num_iis,
                delta=delta,
                random_state=rng,
            )
        truth = np.array(inconsistent, dtype=int) if inconsistent else None
        return A, np.asarray(b, dtype=float), truth

    return builder


def _truth_reference(
    truth: Optional[np.ndarray],
    qr_indices: np.ndarray,
    top_k: int,
) -> Tuple[Set[int], int, bool]:
    qr_slice = np.asarray(qr_indices[:top_k], dtype=int)
    if truth is None:
        truth_slice = np.asarray([], dtype=int)
    else:
        truth_slice = np.asarray(truth, dtype=int).ravel()

    truth_set = set(truth_slice.tolist())
    qr_set = set(qr_slice.tolist())

    if truth_set:
        qr_hit = truth_set <= qr_set
    else:
        qr_hit = not qr_set

    return truth_set, len(truth_set), qr_hit


def _hit_status(truth_set: Set[int], indices: np.ndarray, top_k: int) -> bool:
    candidate = np.asarray(indices[:top_k], dtype=int)
    candidate_set = set(candidate.tolist())
    if truth_set:
        return truth_set <= candidate_set
    return not candidate_set


def _total_time(log) -> float:
    return float(log.time_sketch + log.time_factor + log.time_iterate)


def run_case(
    spec: ProblemSpec,
    base_sketch_options: SketchSolverOptions,
    base_fossils_options: SketchSolverOptions,
    base_analysis: AnalysisConfig,
) -> dict:
    rng = np.random.default_rng(spec.seed)
    A, b, truth = spec.builder(rng)

    qr_start = time.perf_counter()
    qr_output = Solver.analyze_system_with_qr(A, b)
    if qr_output is None:
        raise RuntimeError("QR solver failed to analyze the system.")
    qr_indices, qr_residual = qr_output
    qr_time = time.perf_counter() - qr_start
    qr_norm = norm(qr_residual)

    truth_set, truth_count, qr_hit = _truth_reference(truth, qr_indices, spec.top_k)

    conflict_budget = max(spec.top_k, truth_count)

    sketch_options = SketchSolverOptions(
        mode=base_sketch_options.mode,
        sketch_method=base_sketch_options.sketch_method,
        sampling_factor=base_sketch_options.sampling_factor,
        sketch_size=spec.sketch_size or base_sketch_options.sketch_size,
        sparsity=base_sketch_options.sparsity,
        regularization=base_sketch_options.regularization,
        rank_tol=base_sketch_options.rank_tol,
        lsqr_tol=base_sketch_options.lsqr_tol,
        lsqr_iter_lim=base_sketch_options.lsqr_iter_lim,
        random_state=base_sketch_options.random_state,
        warm_start=base_sketch_options.warm_start,
    )
    sketch_analysis = AnalysisConfig(
        residual_tol_rel=base_analysis.residual_tol_rel,
        residual_tol_abs=base_analysis.residual_tol_abs,
        top_k_conflicts=conflict_budget,
    )

    sketch_result = analyze_linear_system(A, b, sketch_options, sketch_analysis)
    sketch_hit = _hit_status(truth_set, sketch_result.conflicting_indices, spec.top_k)

    fossils_options = SketchSolverOptions(
        mode=base_fossils_options.mode,
        sketch_method=base_fossils_options.sketch_method,
        sampling_factor=base_fossils_options.sampling_factor,
        sketch_size=spec.sketch_size or base_fossils_options.sketch_size,
        sparsity=base_fossils_options.sparsity,
        regularization=base_fossils_options.regularization,
        rank_tol=base_fossils_options.rank_tol,
        lsqr_tol=base_fossils_options.lsqr_tol,
        lsqr_iter_lim=base_fossils_options.lsqr_iter_lim,
        random_state=base_fossils_options.random_state,
        warm_start=base_fossils_options.warm_start,
    )
    fossils_analysis = AnalysisConfig(
        residual_tol_rel=base_analysis.residual_tol_rel,
        residual_tol_abs=base_analysis.residual_tol_abs,
        top_k_conflicts=conflict_budget,
    )

    fossils_result = analyze_linear_system(A, b, fossils_options, fossils_analysis)
    fossils_hit = _hit_status(truth_set, fossils_result.conflicting_indices, spec.top_k)

    return {
        "spec": spec,
        "truth": sorted(truth_set),
        "truth_count": truth_count,
        "qr_hit": qr_hit,
        "sketch_hit": sketch_hit,
        "fossils_hit": fossils_hit,
        "qr_time": qr_time,
        "sketch_time": _total_time(sketch_result.solver_log),
        "fossils_time": _total_time(fossils_result.solver_log),
        "qr_residual_norm": qr_norm,
        "sketch_residual_norm": sketch_result.residual_norm,
        "fossils_residual_norm": fossils_result.residual_norm,
    }


def _format_ms(seconds: float) -> str:
    return f"{1e3 * seconds:7.2f}"


def run_benchmark() -> None:
    specs = [
        ProblemSpec(
            name="loop_consistent_4000",
            builder=_circular_builder(num_constraints=4000, num_iis=0, delta=1e-3),
            top_k=0,
            seed=0,
        ),
        ProblemSpec(
            name="loop_multi_iis_4800",
            builder=_circular_builder(num_constraints=4800, num_iis=3, delta=1e-3),
            top_k=6,
            seed=1,
        ),
        ProblemSpec(
            name="two_var_multi_iis_6000x1800",
            builder=_two_var_builder(
                num_constraints=6000,
                num_variables=1800,
                num_iis=4,
                delta=5e-4,
            ),
            top_k=8,
            seed=2,
        ),
        ProblemSpec(
            name="midpoint_mix_4500x1500",
            builder=_midpoint_builder(
                num_constraints=4500,
                num_variables=1500,
                num_iis=3,
                delta=7.5e-4,
            ),
            top_k=8,
            seed=3,
        ),
        ProblemSpec(
            name="random_sparse_multi_iis_8000x2400",
            builder=_random_sparse_builder(
                num_constraints=8000,
                num_variables=2400,
                num_iis=5,
                delta=1e-3,
            ),
            top_k=10,
            seed=4,
        ),
    ]

    base_sketch = SketchSolverOptions(
        mode="solve",
        sketch_method="sparse_sign",
        sampling_factor=2.5,
        sparsity=6,
        random_state=42,
        regularization=0.0,
        rank_tol=1e-10,
    )
    base_fossils = SketchSolverOptions(
        mode="precondition",
        sketch_method="sparse_sign",
        sampling_factor=3.0,
        sparsity=6,
        lsqr_tol=1e-7,
        lsqr_iter_lim=150,
        random_state=123,
        rank_tol=1e-10,
    )
    base_analysis = AnalysisConfig(
        residual_tol_rel=1e-7,
        residual_tol_abs=1e-8,
        top_k_conflicts=10,
    )

    header = (
        "Case".ljust(28)
        + " | #Conf | QR  | Sketch | FOSSILS | QR (ms) | Sketch (ms) | Fossils (ms)"
    )
    print(header)
    print("-" * len(header))

    for spec in specs:
        result = run_case(spec, base_sketch, base_fossils, base_analysis)
        qr_status = "OK" if result["qr_hit"] else "MISS"
        sketch_status = "OK" if result["sketch_hit"] else "MISS"
        fossils_status = "OK" if result["fossils_hit"] else "MISS"
        print(
            f"{spec.name.ljust(28)} | {result['truth_count']:5d}"
            f" | {qr_status:>3} | {sketch_status:>6} | {fossils_status:>7}"
            f" | {_format_ms(result['qr_time'])} | {_format_ms(result['sketch_time'])}"
            f" | {_format_ms(result['fossils_time'])}"
        )

    print("\nLegend: OK means every ground-truth conflicting constraint appeared within the top_k candidates.")


if __name__ == "__main__":  # pragma: no cover
    run_benchmark()
