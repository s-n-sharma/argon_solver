#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "SuiteSparseQR_C.h"
#include "cholmod.h"

#include "qr_conflict_analyzer.hpp"

extern "C" {
void dgeqp3_(int* m, int* n, double* a, int* lda, int* jpvt, double* tau, double* work, int* lwork, int* info);
void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* a, int* lda, double* tau, double* c, int* ldc, double* work, int* lwork, int* info);
void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* iwork, int* info);
}

struct BenchmarkRow {
    std::string generator;
    SuiteSparse_long size;
    SuiteSparse_long nnz;
    double sparse_ms;
    double dense_ms;
    double svd_ms;
    bool sparse_detected;
    bool dense_detected;
    bool svd_detected;
};

namespace {

cholmod_sparse* generate_random_sparse(
    SuiteSparse_long m,
    SuiteSparse_long n,
    double density,
    SuiteSparse_long seed,
    cholmod_common* common
) {
    SuiteSparse_long nnz = std::max<SuiteSparse_long>(1, static_cast<SuiteSparse_long>(density * static_cast<double>(m) * static_cast<double>(n)));
    cholmod_triplet* triplet = cholmod_l_allocate_triplet(m, n, nnz, 0, CHOLMOD_REAL, common);
    if (!triplet) {
        throw std::runtime_error("Failed to allocate random triplet");
    }

    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::uniform_int_distribution<SuiteSparse_long> row_dist(0, m - 1);
    std::uniform_int_distribution<SuiteSparse_long> col_dist(0, n - 1);
    std::normal_distribution<double> value_dist(0.0, 1.0);

    SuiteSparse_long* Ti = static_cast<SuiteSparse_long*>(triplet->i);
    SuiteSparse_long* Tj = static_cast<SuiteSparse_long*>(triplet->j);
    double* Tx = static_cast<double*>(triplet->x);

    for (SuiteSparse_long k = 0; k < nnz; ++k) {
        Ti[k] = row_dist(rng);
        Tj[k] = col_dist(rng);
        Tx[k] = value_dist(rng);
    }
    triplet->nnz = nnz;

    cholmod_sparse* A = cholmod_l_triplet_to_sparse(triplet, nnz, common);
    cholmod_l_free_triplet(&triplet, common);
    if (!A) {
        throw std::runtime_error("Failed to convert random triplet to sparse matrix");
    }
    return A;
}

cholmod_sparse* generate_circular_graph(
    SuiteSparse_long m,
    SuiteSparse_long n,
    cholmod_common* common
) {
    cholmod_triplet* triplet = cholmod_l_allocate_triplet(m, n, 2 * m, 0, CHOLMOD_REAL, common);
    if (!triplet) {
        throw std::runtime_error("Failed to allocate circular graph triplet");
    }
    SuiteSparse_long* Ti = static_cast<SuiteSparse_long*>(triplet->i);
    SuiteSparse_long* Tj = static_cast<SuiteSparse_long*>(triplet->j);
    double* Tx = static_cast<double*>(triplet->x);

    for (SuiteSparse_long row = 0; row < m; ++row) {
        Ti[2 * row] = row;
        Tj[2 * row] = row % n;
        Tx[2 * row] = 1.0;

        Ti[2 * row + 1] = row;
        Tj[2 * row + 1] = (row + 1) % n;
        Tx[2 * row + 1] = -1.0;
    }
    triplet->nnz = 2 * m;

    cholmod_sparse* A = cholmod_l_triplet_to_sparse(triplet, triplet->nnz, common);
    cholmod_l_free_triplet(&triplet, common);
    if (!A) {
        throw std::runtime_error("Failed to convert circular graph triplet to sparse matrix");
    }
    return A;
}

cholmod_sparse* generate_block_sparse_diagonal(
    SuiteSparse_long m,
    SuiteSparse_long n,
    SuiteSparse_long block,
    cholmod_common* common
) {
    cholmod_triplet* triplet = cholmod_l_allocate_triplet(m, n, m, 0, CHOLMOD_REAL, common);
    if (!triplet) {
        throw std::runtime_error("Failed to allocate block-diagonal triplet");
    }
    SuiteSparse_long* Ti = static_cast<SuiteSparse_long*>(triplet->i);
    SuiteSparse_long* Tj = static_cast<SuiteSparse_long*>(triplet->j);
    double* Tx = static_cast<double*>(triplet->x);

    double val = 1.0;
    for (SuiteSparse_long row = 0; row < m; ++row) {
        SuiteSparse_long col = (row / block) % n;
        Ti[row] = row;
        Tj[row] = col;
        Tx[row] = val;
        val = -val;
    }
    triplet->nnz = m;

    cholmod_sparse* A = cholmod_l_triplet_to_sparse(triplet, triplet->nnz, common);
    cholmod_l_free_triplet(&triplet, common);
    if (!A) {
        throw std::runtime_error("Failed to convert block-diagonal triplet to sparse matrix");
    }
    return A;
}

cholmod_dense* make_inconsistent_rhs(
    cholmod_sparse* A,
    double noise_scale,
    std::mt19937_64& rng,
    cholmod_common* common
) {
    const SuiteSparse_long m = A->nrow;
    const SuiteSparse_long n = A->ncol;

    cholmod_dense* x = cholmod_l_allocate_dense(n, 1, n, CHOLMOD_REAL, common);
    cholmod_dense* b = cholmod_l_allocate_dense(m, 1, m, CHOLMOD_REAL, common);
    if (!x || !b) {
        cholmod_l_free_dense(&x, common);
        cholmod_l_free_dense(&b, common);
        throw std::runtime_error("Failed to allocate dense vectors for RHS generation");
    }

    std::normal_distribution<double> dist(0.0, 1.0);
    double* xdata = static_cast<double*>(x->x);
    for (SuiteSparse_long i = 0; i < n; ++i) {
        xdata[i] = dist(rng);
    }

    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    if (!cholmod_l_sdmult(A, 0, alpha, beta, x, b, common)) {
        cholmod_l_free_dense(&x, common);
        cholmod_l_free_dense(&b, common);
        throw std::runtime_error("Failed to compute A * x for RHS generation");
    }
    cholmod_l_free_dense(&x, common);

    double* bdata = static_cast<double*>(b->x);
    const double perturb = noise_scale * (10.0 + std::abs(bdata[m - 1]));
    bdata[m - 1] += perturb;
    return b;
}

struct DenseDetectionResult {
    double milliseconds;
    bool detected;
};

DenseDetectionResult dense_conflict_detection(
    cholmod_sparse* A,
    const cholmod_dense* b,
    cholmod_common* common
) {
    const int m = static_cast<int>(A->nrow);
    const int n = static_cast<int>(A->ncol);
    const int lda = m;
    const int k = std::min(m, n);

    cholmod_dense* Adense = cholmod_l_sparse_to_dense(A, common);
    if (!Adense) {
        throw std::runtime_error("Failed to convert sparse matrix to dense");
    }
    const double* Adata_ptr = static_cast<const double*>(Adense->x);
    const SuiteSparse_long ld = Adense->d;

    std::vector<double> Adata(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            Adata[static_cast<std::size_t>(col) * lda + row] = Adata_ptr[static_cast<std::size_t>(col) * ld + row];
        }
    }
    cholmod_l_free_dense(&Adense, common);

    std::vector<double> rhs(static_cast<std::size_t>(m));
    const double* bdata = static_cast<const double*>(b->x);
    for (int row = 0; row < m; ++row) {
        rhs[static_cast<std::size_t>(row)] = bdata[row];
    }

    std::vector<double> tau(static_cast<std::size_t>(k));
    std::vector<int> jpvt(static_cast<std::size_t>(n), 0);

    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgeqp3_(const_cast<int*>(&m), const_cast<int*>(&n), Adata.data(), const_cast<int*>(&lda), jpvt.data(), tau.data(), &work_query, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dgeqp3 work size query failed");
    }
    lwork = static_cast<int>(work_query);
    std::vector<double> work(static_cast<std::size_t>(std::max(lwork, 1)));

    auto start = std::chrono::steady_clock::now();
    dgeqp3_(const_cast<int*>(&m), const_cast<int*>(&n), Adata.data(), const_cast<int*>(&lda), jpvt.data(), tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dgeqp3 factorization failed");
    }

    double max_diag = 0.0;
    for (int i = 0; i < k; ++i) {
        const double diag = std::abs(Adata[static_cast<std::size_t>(i) * lda + i]);
        max_diag = std::max(max_diag, diag);
    }
    const double tol = max_diag * 1e-10;
    int rank = 0;
    for (int i = 0; i < k; ++i) {
        const double diag = std::abs(Adata[static_cast<std::size_t>(i) * lda + i]);
        if (diag > tol) {
            ++rank;
        }
    }

    int nrhs = 1;
    int ldc = m;
    lwork = -1;
    double work_q2 = 0.0;
    char side = 'L';
    char trans = 'T';
    dormqr_(&side, &trans, const_cast<int*>(&m), &nrhs, const_cast<int*>(&k), Adata.data(), const_cast<int*>(&lda), tau.data(), rhs.data(), &ldc, &work_q2, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dormqr work size query failed");
    }
    lwork = static_cast<int>(work_q2);
    std::vector<double> work2(static_cast<std::size_t>(std::max(lwork, 1)));
    dormqr_(&side, &trans, const_cast<int*>(&m), &nrhs, const_cast<int*>(&k), Adata.data(), const_cast<int*>(&lda), tau.data(), rhs.data(), &ldc, work2.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dormqr application failed");
    }
    auto end = std::chrono::steady_clock::now();

    double residual_norm_sq = 0.0;
    for (int i = rank; i < m; ++i) {
        const double value = rhs[static_cast<std::size_t>(i)];
        residual_norm_sq += value * value;
    }
    const bool detected = residual_norm_sq > 1e-18;
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();

    return {milliseconds, detected};
}

struct SVDDetectionResult {
    double milliseconds;
    bool detected;
};

SVDDetectionResult svd_conflict_detection(
    cholmod_sparse* A,
    const cholmod_dense* b,
    cholmod_common* common
) {
    const int m = static_cast<int>(A->nrow);
    const int n = static_cast<int>(A->ncol);
    const int lda = m;
    const int k = std::min(m, n);

    cholmod_dense* Adense = cholmod_l_sparse_to_dense(A, common);
    if (!Adense) {
        throw std::runtime_error("Failed to convert sparse matrix to dense for SVD");
    }
    const double* Adata_ptr = static_cast<const double*>(Adense->x);
    const SuiteSparse_long ld = Adense->d;

    std::vector<double> Adata(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            Adata[static_cast<std::size_t>(col) * lda + row] = Adata_ptr[static_cast<std::size_t>(col) * ld + row];
        }
    }
    cholmod_l_free_dense(&Adense, common);

    std::vector<double> singular_values(static_cast<std::size_t>(k));
    std::vector<double> U(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    std::vector<double> VT(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    std::vector<double> Aorig = Adata;

    std::vector<double> bvec(static_cast<std::size_t>(m));
    const double* bdata = static_cast<const double*>(b->x);
    for (int row = 0; row < m; ++row) {
        bvec[static_cast<std::size_t>(row)] = bdata[row];
    }

    int info = 0;
    char jobz = 'S';
    int ldu = m;
    int ldvt = k;
    int lwork = -1;
    double work_query = 0.0;
    std::vector<int> iwork(static_cast<std::size_t>(std::max(1, 8 * k)));

    dgesdd_(&jobz, const_cast<int*>(&m), const_cast<int*>(&n), Adata.data(), const_cast<int*>(&lda), singular_values.data(), U.data(), &ldu, VT.data(), &ldvt, &work_query, &lwork, iwork.data(), &info);
    if (info != 0) {
        throw std::runtime_error("dgesdd work size query failed");
    }
    lwork = static_cast<int>(work_query);
    std::vector<double> work(static_cast<std::size_t>(std::max(lwork, 1)));

    auto start = std::chrono::steady_clock::now();
    dgesdd_(&jobz, const_cast<int*>(&m), const_cast<int*>(&n), Adata.data(), const_cast<int*>(&lda), singular_values.data(), U.data(), &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
    if (info != 0) {
        throw std::runtime_error("dgesdd factorization failed");
    }

    double sigma_max = 0.0;
    for (double s : singular_values) {
        sigma_max = std::max(sigma_max, s);
    }
    const double tol = sigma_max * 1e-10;
    int rank = 0;
    for (double s : singular_values) {
        if (s > tol) {
            ++rank;
        }
    }

    std::vector<double> y(static_cast<std::size_t>(k), 0.0);
    for (int i = 0; i < k; ++i) {
        double dot = 0.0;
        for (int row = 0; row < m; ++row) {
            dot += U[static_cast<std::size_t>(row) + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldu)] * bvec[static_cast<std::size_t>(row)];
        }
        y[static_cast<std::size_t>(i)] = dot;
    }

    std::vector<double> coeff(static_cast<std::size_t>(k), 0.0);
    for (int i = 0; i < k; ++i) {
        if (singular_values[static_cast<std::size_t>(i)] > tol) {
            coeff[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(i)] / singular_values[static_cast<std::size_t>(i)];
        }
    }

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            sum += VT[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldvt)] * coeff[static_cast<std::size_t>(i)];
        }
        x[static_cast<std::size_t>(j)] = sum;
    }

    std::vector<double> residual(static_cast<std::size_t>(m), 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            residual[static_cast<std::size_t>(row)] += Aorig[static_cast<std::size_t>(col) * static_cast<std::size_t>(lda) + row] * x[static_cast<std::size_t>(col)];
        }
    }
    for (int row = 0; row < m; ++row) {
        residual[static_cast<std::size_t>(row)] = bvec[static_cast<std::size_t>(row)] - residual[static_cast<std::size_t>(row)];
    }
    auto end = std::chrono::steady_clock::now();

    double residual_norm_sq = 0.0;
    for (double value : residual) {
        residual_norm_sq += value * value;
    }

    const bool detected = residual_norm_sq > 1e-18;
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();

    return {milliseconds, detected};
}

} // namespace

int main() {
    cholmod_common common;
    cholmod_l_start(&common);

    std::vector<SuiteSparse_long> sizes = {5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000};
    std::vector<BenchmarkRow> results;
    results.reserve(sizes.size() * 3);

    std::mt19937_64 rhs_rng(2025);

    struct GeneratorEntry {
        std::string name;
        std::function<cholmod_sparse*(SuiteSparse_long, SuiteSparse_long, SuiteSparse_long, cholmod_common*)> make;
    };

    std::vector<GeneratorEntry> generators;
    generators.push_back({
        "random_sparse",
        [](SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long seed, cholmod_common* common_ptr) {
            return generate_random_sparse(m, n, 0.01, seed, common_ptr);
        }
    });
    generators.push_back({
        "circular_graph",
        [](SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long /*seed*/, cholmod_common* common_ptr) {
            return generate_circular_graph(m, n, common_ptr);
        }
    });
    generators.push_back({
        "block_diagonal",
        [](SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long /*seed*/, cholmod_common* common_ptr) {
            SuiteSparse_long block = std::max<SuiteSparse_long>(1, n / 10);
            return generate_block_sparse_diagonal(m, n, block, common_ptr);
        }
    });

    for (const auto& entry : generators) {
        for (SuiteSparse_long size : sizes) {
            const SuiteSparse_long m = size;
            const SuiteSparse_long n = std::max<SuiteSparse_long>(2, size / 2);
            cholmod_sparse* A = entry.make(m, n, 42, &common);
            if (!A) {
                throw std::runtime_error("Matrix generation failed");
            }

            cholmod_dense* b = make_inconsistent_rhs(A, 1.0, rhs_rng, &common);
            if (!b) {
                cholmod_l_free_sparse(&A, &common);
                throw std::runtime_error("Failed to create inconsistent RHS");
            }

            auto sparse_start = std::chrono::steady_clock::now();
            QRConflictAnalysisResult sparse_analysis = analyze_system_with_qr(
                A,
                b,
                1e-12,
                1e-9,
                &common,
                false
            );
            auto sparse_end = std::chrono::steady_clock::now();
            const double sparse_ms = std::chrono::duration<double, std::milli>(sparse_end - sparse_start).count();

            DenseDetectionResult dense_result = dense_conflict_detection(A, b, &common);
            SVDDetectionResult svd_result = svd_conflict_detection(A, b, &common);

            SuiteSparse_long nnz = cholmod_l_nnz(A, &common);
            results.push_back({
                entry.name,
                size,
                nnz,
                sparse_ms,
                dense_result.milliseconds,
                svd_result.milliseconds,
                sparse_analysis.isConflicting,
                dense_result.detected,
                svd_result.detected,
            });

            cholmod_l_free_dense(&b, &common);
            cholmod_l_free_sparse(&A, &common);
        }
    }

    cholmod_l_finish(&common);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "generator,size,nnz,sparse_ms,dense_ms,svd_ms,sparse_detected,dense_detected,svd_detected\n";
    for (const auto& row : results) {
        std::cout << row.generator << ','
                  << row.size << ','
                  << row.nnz << ','
                  << row.sparse_ms << ','
                  << row.dense_ms << ','
                  << row.svd_ms << ','
                  << static_cast<int>(row.sparse_detected) << ','
                  << static_cast<int>(row.dense_detected) << ','
                  << static_cast<int>(row.svd_detected) << '\n';
    }

    return 0;
}
