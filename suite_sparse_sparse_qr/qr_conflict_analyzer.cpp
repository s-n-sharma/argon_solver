#include "qr_conflict_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

std::vector<double> extractColumnMajorVector(const cholmod_dense* matrix) {
    if (!matrix || matrix->ncol != 1) {
        throw std::runtime_error("Expected a column vector");
    }
    const auto length = static_cast<std::size_t>(matrix->nrow);
    const double* data = static_cast<const double*>(matrix->x);
    return std::vector<double>(data, data + length);
}

std::vector<double> backSubstituteUpper(
    const cholmod_dense* Rdense,
    SuiteSparse_long rank,
    const std::vector<double>& rhs
) {
    if (!Rdense) {
        throw std::runtime_error("Missing R factor");
    }
    if (static_cast<SuiteSparse_long>(rhs.size()) < rank) {
        throw std::runtime_error("Right-hand side shorter than rank");
    }
    if (rank == 0) {
        return {};
    }

    const double* Rdata = static_cast<const double*>(Rdense->x);
    const SuiteSparse_long leadingDim = Rdense->d;
    std::vector<double> y(static_cast<std::size_t>(rank), 0.0);

    for (SuiteSparse_long row = rank - 1; row >= 0; --row) {
        double sum = 0.0;
        for (SuiteSparse_long col = row + 1; col < rank; ++col) {
            const double entry = Rdata[col * leadingDim + row];
            sum += entry * y[static_cast<std::size_t>(col)];
        }
        const double diag = Rdata[row * leadingDim + row];
        if (std::abs(diag) < 1e-14) {
            throw std::runtime_error("Encountered near-zero diagonal in R");
        }
        const double rhsValue = rhs[static_cast<std::size_t>(row)];
        y[static_cast<std::size_t>(row)] = (rhsValue - sum) / diag;
    }
    return y;
}

} // namespace

QRConflictAnalysisResult analyze_system_with_qr(
    cholmod_sparse* A,
    cholmod_dense* b,
    double factorTolerance,
    double conflictThreshold,
    cholmod_common* common,
    bool verbose
) {
    if (!A || !b || !common) {
        throw std::invalid_argument("Null pointer input to analyze_system_with_qr");
    }

    QRConflictAnalysisResult result{};

    cholmod_sparse* Q = nullptr;
    cholmod_sparse* R = nullptr;
    SuiteSparse_long* E = nullptr;

    if (b->ncol != 1) {
        throw std::invalid_argument("Only single right-hand sides are supported");
    }
    if (A->nrow != b->nrow) {
        throw std::invalid_argument("Dimension mismatch between A and b");
    }

    const SuiteSparse_long econ = static_cast<SuiteSparse_long>(A->nrow);
    // Factor A to obtain Q, R, and the column permutation E.
    const int64_t rankEstimate = SuiteSparseQR_C_QR(
        SPQR_ORDERING_DEFAULT,
        factorTolerance,
        econ,
        A,
        &Q,
        &R,
        &E,
        common
    );

    if (rankEstimate < 0) {
        throw std::runtime_error("SuiteSparseQR_C_QR failed");
    }

    const SuiteSparse_long rank = static_cast<SuiteSparse_long>(rankEstimate);
    result.rank = rank;

    // Project b into the Q basis.
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cholmod_dense* cDense = cholmod_l_allocate_dense(Q->ncol, b->ncol, Q->ncol, CHOLMOD_REAL, common);
    if (!cDense) {
        cholmod_l_free_sparse(&Q, common);
        cholmod_l_free_sparse(&R, common);
        SuiteSparse_free(E);
        throw std::runtime_error("Failed to allocate projection vector");
    }

    if (!cholmod_l_sdmult(Q, 1, alpha, beta, b, cDense, common)) {
        cholmod_l_free_dense(&cDense, common);
        cholmod_l_free_sparse(&Q, common);
        cholmod_l_free_sparse(&R, common);
        SuiteSparse_free(E);
        throw std::runtime_error("Q^T * b computation failed");
    }

    std::vector<double> cVector = extractColumnMajorVector(cDense);

    double conflictNormSq = 0.0;
    for (std::size_t idx = static_cast<std::size_t>(rank); idx < cVector.size(); ++idx) {
        conflictNormSq += cVector[idx] * cVector[idx];
    }
    result.conflictNorm = std::sqrt(conflictNormSq);
    result.isUnderconstrained = rank < static_cast<SuiteSparse_long>(A->ncol);
    result.isConflicting = result.conflictNorm > conflictThreshold;

    if (verbose) {
        std::cout << "Computed rank: " << rank << '\n';
        std::cout << "Conflict norm: " << result.conflictNorm << '\n';
    }

    std::vector<double> c1;
    c1.reserve(static_cast<std::size_t>(rank));
    for (SuiteSparse_long i = 0; i < rank; ++i) {
        c1.push_back(cVector[static_cast<std::size_t>(i)]);
    }

    cholmod_dense* Rdense = cholmod_l_sparse_to_dense(R, common);
    if (!Rdense) {
        cholmod_l_free_dense(&cDense, common);
        cholmod_l_free_sparse(&Q, common);
        cholmod_l_free_sparse(&R, common);
        SuiteSparse_free(E);
        throw std::runtime_error("Failed to convert R to dense form");
    }

    if (verbose) {
        std::cout << "R diagonal entries:" << '\n';
        const double* Rdata = static_cast<const double*>(Rdense->x);
        const SuiteSparse_long ld = Rdense->d;
        const SuiteSparse_long diagCount = std::min<SuiteSparse_long>(rank, Rdense->nrow);
        for (SuiteSparse_long i = 0; i < diagCount; ++i) {
            std::cout << "  R[" << i << "," << i << "] = "
                      << Rdata[i * ld + i] << '\n';
        }
    }

    std::vector<double> y1 = backSubstituteUpper(Rdense, rank, c1);

    if (verbose) {
        std::cout << "Top-left R block solution (y1):" << '\n';
        for (SuiteSparse_long i = 0; i < rank; ++i) {
            std::cout << "  y1[" << i << "] = " << y1[static_cast<std::size_t>(i)] << '\n';
        }
    }

    const SuiteSparse_long ncols = A->ncol;
    std::vector<double> xHat(static_cast<std::size_t>(ncols), 0.0);
    for (SuiteSparse_long i = 0; i < rank; ++i) {
        const SuiteSparse_long permIndex = E ? E[i] : i;
        if (permIndex < 0 || permIndex >= ncols) {
            cholmod_l_free_dense(&Rdense, common);
            cholmod_l_free_dense(&cDense, common);
            cholmod_l_free_sparse(&Q, common);
            cholmod_l_free_sparse(&R, common);
            SuiteSparse_free(E);
            throw std::runtime_error("Permutation index out of range");
        }
        xHat[static_cast<std::size_t>(permIndex)] = y1[static_cast<std::size_t>(i)];
    }

    cholmod_dense* xDense = cholmod_l_allocate_dense(ncols, 1, ncols, CHOLMOD_REAL, common);
    if (!xDense) {
        cholmod_l_free_dense(&Rdense, common);
        cholmod_l_free_dense(&cDense, common);
        cholmod_l_free_sparse(&Q, common);
        cholmod_l_free_sparse(&R, common);
        SuiteSparse_free(E);
        throw std::runtime_error("Failed to allocate xDense");
    }
    double* xData = static_cast<double*>(xDense->x);
    std::copy(xHat.begin(), xHat.end(), xData);

    cholmod_dense* Ax = cholmod_l_allocate_dense(A->nrow, 1, A->nrow, CHOLMOD_REAL, common);
    if (!Ax) {
        cholmod_l_free_dense(&xDense, common);
        cholmod_l_free_dense(&Rdense, common);
        cholmod_l_free_dense(&cDense, common);
        cholmod_l_free_sparse(&Q, common);
        cholmod_l_free_sparse(&R, common);
        SuiteSparse_free(E);
        throw std::runtime_error("Failed to allocate Ax");
    }

    double alphaAx[2] = {1.0, 0.0};
    double betaAx[2] = {0.0, 0.0};

    // Compute residual r = b - A x_hat.
    if (!cholmod_l_sdmult(A, 0, alphaAx, betaAx, xDense, Ax, common)) {
        cholmod_l_free_dense(&Ax, common);
        cholmod_l_free_dense(&xDense, common);
        cholmod_l_free_dense(&Rdense, common);
        cholmod_l_free_dense(&cDense, common);
        cholmod_l_free_sparse(&Q, common);
        cholmod_l_free_sparse(&R, common);
        SuiteSparse_free(E);
        throw std::runtime_error("A * x computation failed");
    }

    const double* bData = static_cast<const double*>(b->x);
    const double* AxData = static_cast<const double*>(Ax->x);
    const SuiteSparse_long nrows = A->nrow;

    result.residual.resize(static_cast<std::size_t>(nrows));
    for (SuiteSparse_long i = 0; i < nrows; ++i) {
        result.residual[static_cast<std::size_t>(i)] = bData[i] - AxData[i];
    }

    result.sortedIndices.resize(static_cast<std::size_t>(nrows));
    std::iota(result.sortedIndices.begin(), result.sortedIndices.end(), static_cast<SuiteSparse_long>(0));
    std::sort(
        result.sortedIndices.begin(),
        result.sortedIndices.end(),
        [&result](SuiteSparse_long lhs, SuiteSparse_long rhs) {
            const double left = std::abs(result.residual[static_cast<std::size_t>(lhs)]);
            const double right = std::abs(result.residual[static_cast<std::size_t>(rhs)]);
            if (left == right) {
                return lhs < rhs;
            }
            return left > right;
        }
    );

    if (verbose) {
        std::cout << "Residuals (sorted indices):\n";
        for (SuiteSparse_long idx : result.sortedIndices) {
            const double value = result.residual[static_cast<std::size_t>(idx)];
            std::cout << "  index " << idx << ": " << value << '\n';
        }
    }

    cholmod_l_free_dense(&Ax, common);
    cholmod_l_free_dense(&xDense, common);
    cholmod_l_free_dense(&Rdense, common);
    cholmod_l_free_dense(&cDense, common);
    cholmod_l_free_sparse(&Q, common);
    cholmod_l_free_sparse(&R, common);
    SuiteSparse_free(E);

    return result;
}
