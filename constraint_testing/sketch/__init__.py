"""Sketch-and-solve algorithms for constraint diagnostics."""

from .solver import SketchConfig, SketchResult, analyze_system_with_sketch  # noqa: F401
from .sketching import SketchSample, apply_sparse_sketch  # noqa: F401
