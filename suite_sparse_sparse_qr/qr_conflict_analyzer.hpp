#pragma once

#include <vector>
#include <cstdint>
#include "SuiteSparseQR_C.h"
#include "cholmod.h"

struct QRConflictAnalysisResult {
    std::vector<SuiteSparse_long> sortedIndices;
    std::vector<double> residual;
    SuiteSparse_long rank;
    bool isUnderconstrained;
    bool isConflicting;
    double conflictNorm;
};

QRConflictAnalysisResult analyze_system_with_qr(
    cholmod_sparse* A,
    cholmod_dense* b,
    double factorTolerance,
    double conflictThreshold,
    cholmod_common* common,
    bool verbose = false
);
