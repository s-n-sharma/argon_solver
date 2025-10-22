from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as spa
from numpy.random import Generator
from scipy.linalg._sketches import cwt_matrix


@dataclass
class SketchSample:
    """Metadata describing the sketch that was applied."""

    method: str
    sketch_matrix: spa.spmatrix
    seed: Optional[int]
    params: dict


def _ensure_csr(A: spa.spmatrix) -> spa.csr_matrix:
    if spa.issparse(A):
        return A.tocsr()
    return spa.csr_matrix(A)


def _prepare_rng(seed: Optional[int | Generator]) -> Generator:
    if isinstance(seed, Generator):
        return seed
    return np.random.default_rng(seed)


def _uniform_sparse_matrix(
    m: int,
    n: int,
    sketch_size: Optional[int],
    rng: Generator,
) -> spa.csr_matrix:
    k = sketch_size if sketch_size is not None else min(m, max(2 * n, 1))
    if k <= 0:
        raise ValueError("sketch_size must be positive")
    if k > m:
        k = m
    rows = np.arange(k)
    cols = rng.integers(0, m, size=k)
    scale = np.sqrt(max(m / k, 1.0))
    data = np.full(k, scale, dtype=float)
    return spa.csr_matrix((data, (rows, cols)), shape=(k, m))


def _clarkson_woodruff_matrix(
    m: int,
    n: int,
    sketch_size: Optional[int],
    rng: Generator,
) -> spa.csc_matrix:
    k = sketch_size if sketch_size is not None else n
    if k <= 0:
        raise ValueError("sketch_size must be positive")
    if k > m:
        k = m
    S = cwt_matrix(k, m, rng=rng)
    return S.tocsc()


def _sparse_sign_matrix(
    m: int,
    n: int,
    sketch_size: Optional[int],
    rng: Generator,
    sparsity_parameter: Optional[int],
) -> spa.csc_matrix:
    d = sketch_size if sketch_size is not None else max(20 * n, n + 4)
    d = max(1, d)
    zeta = sparsity_parameter if sparsity_parameter is not None else min(d, 8)
    zeta = int(max(2, min(zeta, d)))
    cols = np.repeat(np.arange(m), zeta)
    rows = rng.integers(0, d, size=cols.size)
    signs = rng.choice([-1.0, 1.0], size=cols.size)
    data = signs / np.sqrt(zeta)
    return spa.csc_matrix((data, (rows, cols)), shape=(d, m))


def apply_sparse_sketch(
    A: spa.spmatrix,
    b: np.ndarray,
    sketch_size: Optional[int],
    method: str = "sparse_sign",
    seed: Optional[int | Generator] = None,
    sparsity_parameter: Optional[int] = None,
) -> Tuple[spa.csc_matrix, np.ndarray, SketchSample]:
 
    A_csr = _ensure_csr(A)
    m, n = A_csr.shape
    b_vec = np.asarray(b).reshape(-1)
    if b_vec.shape[0] != m:
        raise ValueError("Shapes of A and b do not align.")

    rng = _prepare_rng(seed)
    method_key = method.lower()

    if method_key in {"uniform", "uniform_sparse"}:
        S = _uniform_sparse_matrix(m, n, sketch_size, rng)
        chosen_method = "uniform_sparse"
        params = {"sketch_size": S.shape[0]}
    elif method_key in {"clarkson", "clarkson_woodruff", "cw"}:
        S = _clarkson_woodruff_matrix(m, n, sketch_size, rng)
        chosen_method = "clarkson_woodruff"
        params = {"sketch_size": S.shape[0]}
    elif method_key in {"sparse_sign", "sign"}:
        S = _sparse_sign_matrix(m, n, sketch_size, rng, sparsity_parameter)
        chosen_method = "sparse_sign"
        params = {"sketch_size": S.shape[0], "sparsity_parameter": sparsity_parameter}
    else:
        raise ValueError(f"Unknown sketch method '{method}'.")

    SA = (S @ A_csr).tocsc()
    sb = np.asarray(S @ b_vec)

    sample = SketchSample(
        method=chosen_method,
        sketch_matrix=S,
        seed=None if isinstance(seed, Generator) else seed,
        params=params,
    )

    return SA, sb, sample
