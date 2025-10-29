"""Sketch-based solvers and diagnostics for CAD constraint systems."""

from .analysis import AnalysisConfig, AnalysisResult, analyze_linear_system
from .solvers import (
	SketchSolverOptions,
	SketchAndSolveLSQ,
	SketchAndPreconditionLSQ,
	solve_least_squares,
)
from .sketchers import SketchSample
from .randomized_svd import RandomizedSVDSolver, RandomizedSVDResult

__all__ = [
	"AnalysisConfig",
	"AnalysisResult",
	"SketchSolverOptions",
	"SketchAndSolveLSQ",
	"SketchAndPreconditionLSQ",
	"SketchSample",
	"analyze_linear_system",
	"solve_least_squares",
	"RandomizedSVDSolver",
	"RandomizedSVDResult",
]
