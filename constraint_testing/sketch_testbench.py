"""Benchmark conflict detection for CAD-style constraint systems."""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple

import numpy as np
import scipy.sparse as spa
from numpy.linalg import norm

try:  # Allow running as both a module and a script.
    from .solver_utils import Solver
    from .sketch import SketchConfig, analyze_system_with_sketch
    from .sketch.fossils import (
        FossilsConfig,
        analyze_system_with_fossils,
    )
except ImportError:  # pragma: no cover - fallback for direct execution.
    from solver_utils import Solver
    from sketch import SketchConfig, analyze_system_with_sketch
    from sketch.fossils import FossilsConfig, analyze_system_with_fossils


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


def _random_problem_builder(
    m: int,
    n: int,
    density: float,
    noise: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, np.ndarray]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, np.ndarray]:
        data_rvs = rng.standard_normal
        A = spa.random(m, n, density=density, format="csr", data_rvs=data_rvs)
        x_star = rng.standard_normal(n)
        b = np.asarray(A @ x_star, dtype=float)
        conflict_idx = rng.integers(m)
        b[conflict_idx] += noise
        truth = np.array([int(conflict_idx)], dtype=int)
        return A.tocsc(), b, truth

    return builder


def _length_cycle_builder(
    num_constraints: int,
    delta: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, np.ndarray]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, np.ndarray]:
        if num_constraints < 2:
            raise ValueError("num_constraints must be at least 2")
        base_m = num_constraints - 1
        num_points = base_m
        row_idx = np.repeat(np.arange(base_m), 2)
        col_idx = np.empty(2 * base_m, dtype=int)
        indices = np.arange(base_m)
        col_idx[0::2] = indices
        col_idx[1::2] = (indices + 1) % num_points
        data = np.empty(2 * base_m, dtype=float)
        data[0::2] = -1.0
        data[1::2] = 1.0
        A_base = spa.csr_matrix((data, (row_idx, col_idx)), shape=(base_m, num_points)).tocsc()
        positions = rng.uniform(-1.0, 1.0, size=num_points)
        b = np.asarray(
            positions[(indices + 1) % num_points] - positions[indices],
            dtype=float,
        )
        conflict_idx = rng.integers(base_m)
        conflict_row = A_base.getrow(conflict_idx)
        A = spa.vstack([A_base, conflict_row], format="csc").tocsc()
        b_conflict = b.copy()
        b_aug = np.concatenate([b_conflict, [b_conflict[conflict_idx] + delta]])
        truth = np.array([int(conflict_idx), int(base_m)], dtype=int)
        return A, b_aug, truth

    return builder


def _midpoint_cycle_builder(
    num_constraints: int,
    delta: float,
) -> Callable[[np.random.Generator], Tuple[spa.csc_matrix, np.ndarray, np.ndarray]]:
    def builder(rng: np.random.Generator) -> Tuple[spa.csc_matrix, np.ndarray, np.ndarray]:
        if num_constraints < 3:
            raise ValueError("num_constraints must be at least 3")
        base_m = num_constraints - 1
        num_points = base_m
        row_idx = np.repeat(np.arange(base_m), 3)
        col_idx = np.empty(3 * base_m, dtype=int)
        indices = np.arange(base_m)
        col_idx[0::3] = indices
        col_idx[1::3] = (indices + 1) % num_points
        col_idx[2::3] = (indices + 2) % num_points
        data = np.empty(3 * base_m, dtype=float)
        data[0::3] = 1.0
        data[1::3] = -2.0
        data[2::3] = 1.0
        A_base = spa.csr_matrix((data, (row_idx, col_idx)), shape=(base_m, num_points)).tocsc()
        positions = rng.uniform(-1.0, 1.0, size=num_points)
        b = np.asarray(
            positions[indices]
            - 2.0 * positions[(indices + 1) % num_points]
            + positions[(indices + 2) % num_points],
            dtype=float,
        )
        conflict_idx = rng.integers(base_m)
        conflict_row = A_base.getrow(conflict_idx)
        A = spa.vstack([A_base, conflict_row], format="csc").tocsc()
        b_aug = np.concatenate([b, [b[conflict_idx] + delta]])
        truth = np.array([int(conflict_idx), int(base_m)], dtype=int)
        return A, b_aug, truth

    return builder


def _truth_reference(
    truth: Optional[np.ndarray],
    qr_indices: np.ndarray,
    top_k: int,
) -> Tuple[Set[int], int, bool]:
    qr_slice = np.asarray(qr_indices[:top_k], dtype=int)

    if truth is None:
        truth_slice = qr_slice
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


def run_case(
    spec: ProblemSpec,
    base_sketch_config: SketchConfig,
    base_fossils_config: FossilsConfig,
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

    sketch_config = SketchConfig(
        sketch_size=spec.sketch_size or base_sketch_config.sketch_size,
        random_state=base_sketch_config.random_state,
        sketch_method=base_sketch_config.sketch_method,
        sparsity_parameter=base_sketch_config.sparsity_parameter,
        rank_tol=base_sketch_config.rank_tol,
        residual_tol=base_sketch_config.residual_tol,
        top_k_conflicts=max(spec.top_k, base_sketch_config.top_k_conflicts or spec.top_k),
    )

    sketch_result = analyze_system_with_sketch(A, b, sketch_config)
    sketch_hit = _hit_status(truth_set, sketch_result.sorted_indices, spec.top_k)

    fossils_config = FossilsConfig(
        sketch_size=spec.sketch_size or base_fossils_config.sketch_size,
        embedding_oversample=base_fossils_config.embedding_oversample,
        random_state=base_fossils_config.random_state,
        sketch_method=base_fossils_config.sketch_method,
        sparsity_parameter=base_fossils_config.sparsity_parameter,
        rank_tol=base_fossils_config.rank_tol,
        residual_tol=base_fossils_config.residual_tol,
        lsqr_atol=base_fossils_config.lsqr_atol,
        lsqr_btol=base_fossils_config.lsqr_btol,
        lsqr_iter_lim=base_fossils_config.lsqr_iter_lim,
        heavy_ball_iters=base_fossils_config.heavy_ball_iters,
        heavy_ball_alpha=base_fossils_config.heavy_ball_alpha,
        heavy_ball_beta=base_fossils_config.heavy_ball_beta,
        top_k_conflicts=max(spec.top_k, base_fossils_config.top_k_conflicts or spec.top_k),
    )

    fossils_result = analyze_system_with_fossils(A, b, fossils_config)
    fossils_hit = _hit_status(truth_set, fossils_result.sorted_indices, spec.top_k)

    return {
        "spec": spec,
        "truth": sorted(truth_set),
        "truth_count": truth_count,
        "qr_hit": qr_hit,
        "sketch_hit": sketch_hit,
        "fossils_hit": fossils_hit,
        "qr_time": qr_time,
        "sketch_time": sketch_result.solve_time,
        "fossils_time": fossils_result.solve_time,
        "qr_residual_norm": qr_norm,
        "sketch_residual_norm": sketch_result.residual_norm,
        "fossils_residual_norm": fossils_result.residual_norm,
    }


def _format_ms(seconds: float) -> str:
    return f"{1e3 * seconds:7.2f}"


def run_benchmark() -> None:
    specs = [
        ProblemSpec(
            name="random_over_2000x800",
            builder=_random_problem_builder(m=2000, n=800, density=0.01, noise=1e-3),
            top_k=5,
            seed=0,
        ),
        ProblemSpec(
            name="random_over_8000x2000",
            builder=_random_problem_builder(m=8000, n=2000, density=0.005, noise=1e-3),
            top_k=5,
            seed=1,
        ),
        ProblemSpec(
            name="length_cycle_2000",
            builder=_length_cycle_builder(num_constraints=2000, delta=1e-3),
            top_k=4,
            seed=2,
        ),
        ProblemSpec(
            name="length_cycle_10000",
            builder=_length_cycle_builder(num_constraints=10000, delta=1e-3),
            top_k=4,
            seed=3,
        ),
        ProblemSpec(
            name="midpoint_cycle_3000",
            builder=_midpoint_cycle_builder(num_constraints=3000, delta=1e-3),
            top_k=4,
            seed=4,
        ),
        ProblemSpec(
            name="midpoint_cycle_9000",
            builder=_midpoint_cycle_builder(num_constraints=9000, delta=1e-3),
            top_k=4,
            seed=5,
        ),
    ]

    max_top_k = max(spec.top_k for spec in specs)
    base_sketch = SketchConfig(random_state=42, top_k_conflicts=max_top_k)
    base_fossils = FossilsConfig(random_state=123, top_k_conflicts=max_top_k)

    header = (
        "Case".ljust(28)
        + " | #Conf | QR  | Sketch | FOSSILS | QR (ms) | Sketch (ms) | Fossils (ms)"
    )
    print(header)
    print("-" * len(header))

    for spec in specs:
        result = run_case(spec, base_sketch, base_fossils)
        qr_status = "OK" if result["qr_hit"] else "MISS"
        sketch_status = "OK" if result["sketch_hit"] else "MISS"
        fossils_status = "OK" if result["fossils_hit"] else "MISS"
        print(
            f"{spec.name.ljust(28)} | {result['truth_count']:5d}"
            f" | {qr_status:>3} | {sketch_status:>6} | {fossils_status:>7}"
            f" | {_format_ms(result['qr_time'])} | {_format_ms(result['sketch_time'])}"
            f" | {_format_ms(result['fossils_time'])}"
        )

    print("\nLegend: OK means the conflicting constraint appeared within the top_k candidates.")


if __name__ == "__main__":  # pragma: no cover
    run_benchmark()
