#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "SuiteSparseQR_C.h"
#include "cholmod.h"

extern "C" {
void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* a, int* lda, double* tau, double* c, int* ldc, double* work, int* lwork, int* info);
}

namespace {

using Clock = std::chrono::steady_clock;

struct DenseMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> values; // column-major storage
};

struct Dataset {
    DenseMatrix matrix;
    std::vector<double> rhs;
};

struct DenseBaseline {
    int rows = 0;
    int cols = 0;
    std::vector<double> R; // n x n upper-triangular
    std::vector<double> c; // first n entries of Q^T b
};

inline double& entry(DenseMatrix& mat, int row, int col) {
    return mat.values[static_cast<std::size_t>(col) * mat.rows + row];
}

inline double entry(const DenseMatrix& mat, int row, int col) {
    return mat.values[static_cast<std::size_t>(col) * mat.rows + row];
}

std::vector<double> make_random_vector(int n, std::mt19937_64& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> vec(static_cast<std::size_t>(n));
    for (double& v : vec) {
        v = dist(rng);
    }
    return vec;
}

Dataset make_random_sparse_dataset(int total_rows, int n, double density, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> value_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    DenseMatrix mat;
    mat.rows = total_rows;
    mat.cols = n;
    mat.values.assign(static_cast<std::size_t>(total_rows) * n, 0.0);

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < total_rows; ++row) {
            if (coin(rng) < density) {
                entry(mat, row, col) = value_dist(rng);
            }
        }
    }

    std::vector<double> rhs(total_rows, 0.0);
    std::vector<double> x_true = make_random_vector(n, rng);
    for (int row = 0; row < total_rows; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += entry(mat, row, col) * x_true[col];
        }
        rhs[static_cast<std::size_t>(row)] = sum;
    }

    return {std::move(mat), std::move(rhs)};
}

Dataset make_circular_dataset(int total_rows, int n) {
    DenseMatrix mat;
    mat.rows = total_rows;
    mat.cols = n;
    mat.values.assign(static_cast<std::size_t>(total_rows) * n, 0.0);

    for (int row = 0; row < total_rows; ++row) {
        int c0 = row % n;
        int c1 = (row + 1) % n;
        entry(mat, row, c0) = 1.0;
        entry(mat, row, c1) = -1.0;
    }

    std::vector<double> rhs(total_rows, 0.0);
    for (int row = 0; row < total_rows; ++row) {
        rhs[static_cast<std::size_t>(row)] = (row % 7 == 0) ? 1.0 : 0.0;
    }

    return {std::move(mat), std::move(rhs)};
}

Dataset make_block_diagonal_dataset(int total_rows, int n) {
    DenseMatrix mat;
    mat.rows = total_rows;
    mat.cols = n;
    mat.values.assign(static_cast<std::size_t>(total_rows) * n, 0.0);

    const int block = std::max(1, n / 8);
    double sign = 1.0;
    for (int row = 0; row < total_rows; ++row) {
        int col = (row / block) % n;
        entry(mat, row, col) = sign;
        sign = -sign;
    }

    std::vector<double> rhs(total_rows, 0.0);
    for (int row = 0; row < total_rows; ++row) {
        rhs[static_cast<std::size_t>(row)] = (row % 5 == 0) ? 2.0 : -1.0;
    }

    return {std::move(mat), std::move(rhs)};
}

void extract_R(const std::vector<double>& A_factored, int m, int n, std::vector<double>& R_out) {
    const int ld = m;
    R_out.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col && row < n; ++row) {
            R_out[static_cast<std::size_t>(col) * n + row] = A_factored[static_cast<std::size_t>(col) * ld + row];
        }
    }
}

void back_substitute_upper(int n, const std::vector<double>& R, const std::vector<double>& c, std::vector<double>& x) {
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int row = n - 1; row >= 0; --row) {
        double acc = c[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < n; ++col) {
            acc -= R[static_cast<std::size_t>(col) * n + row] * x[static_cast<std::size_t>(col)];
        }
        double diag = R[static_cast<std::size_t>(row) * n + row];
        x[static_cast<std::size_t>(row)] = (std::abs(diag) > 1e-15) ? acc / diag : 0.0;
    }
}

DenseBaseline factorize_dense_baseline(const DenseMatrix& mat, const std::vector<double>& rhs, int base_rows) {
    if (base_rows > mat.rows) {
        throw std::invalid_argument("Base rows exceed matrix rows");
    }

    DenseBaseline baseline;
    baseline.rows = base_rows;
    baseline.cols = mat.cols;

    const int m = base_rows;
    const int n = mat.cols;
    const int lda = m;

    std::vector<double> A_work(static_cast<std::size_t>(m) * n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            A_work[static_cast<std::size_t>(col) * lda + row] = entry(mat, row, col);
        }
    }

    std::vector<double> tau(static_cast<std::size_t>(std::min(m, n)));

    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgeqrf_(const_cast<int*>(&m), const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), &work_query, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dgeqrf workspace query failed");
    }

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeqrf_(const_cast<int*>(&m), const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dgeqrf factorization failed");
    }

    extract_R(A_work, m, n, baseline.R);

    std::vector<double> b_work(static_cast<std::size_t>(m));
    for (int row = 0; row < m; ++row) {
        b_work[static_cast<std::size_t>(row)] = rhs[static_cast<std::size_t>(row)];
    }

    int nrhs = 1;
    int ldc = m;
    lwork = -1;
    work_query = 0.0;
    char side = 'L';
    char trans = 'T';
    dormqr_(&side, &trans, const_cast<int*>(&m), &nrhs, const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), b_work.data(), &ldc, &work_query, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dormqr workspace query failed");
    }

    lwork = std::max(1, static_cast<int>(work_query));
    work.assign(static_cast<std::size_t>(lwork), 0.0);
    dormqr_(&side, &trans, const_cast<int*>(&m), &nrhs, const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), b_work.data(), &ldc, work.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dormqr apply failed");
    }

    baseline.c.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        baseline.c[static_cast<std::size_t>(i)] = (i < m) ? b_work[static_cast<std::size_t>(i)] : 0.0;
    }

    return baseline;
}

double dense_givens_update_time(const DenseMatrix& mat, const std::vector<double>& rhs, const DenseBaseline& base, int base_rows, int update_row_index) {
    const int n = base.cols;

    std::vector<double> R = base.R;
    std::vector<double> c = base.c;
    std::vector<double> new_row(static_cast<std::size_t>(n));
    for (int col = 0; col < n; ++col) {
        new_row[static_cast<std::size_t>(col)] = entry(mat, update_row_index, col);
    }
    double rhs_extra = rhs[static_cast<std::size_t>(update_row_index)];

    auto start = Clock::now();

    for (int j = 0; j < n; ++j) {
        std::size_t diag_idx = static_cast<std::size_t>(j) * n + j;
        double diag = R[diag_idx];
        double w = new_row[static_cast<std::size_t>(j)];
        double r = std::hypot(diag, w);
        if (r == 0.0) {
            continue;
        }
        double c_rot = diag / r;
        double s_rot = w / r;
        R[diag_idx] = r;
        new_row[static_cast<std::size_t>(j)] = 0.0;

        for (int k = j + 1; k < n; ++k) {
            std::size_t idx = static_cast<std::size_t>(k) * n + j;
            double Rjk = R[idx];
            double wk = new_row[static_cast<std::size_t>(k)];
            double temp = c_rot * Rjk + s_rot * wk;
            new_row[static_cast<std::size_t>(k)] = -s_rot * Rjk + c_rot * wk;
            R[idx] = temp;
        }

        double temp_c = c_rot * c[static_cast<std::size_t>(j)] + s_rot * rhs_extra;
        rhs_extra = -s_rot * c[static_cast<std::size_t>(j)] + c_rot * rhs_extra;
        c[static_cast<std::size_t>(j)] = temp_c;
    }

    std::vector<double> solution;
    back_substitute_upper(n, R, c, solution);

    auto end = Clock::now();
    (void)base_rows;
    (void)solution;
    return std::chrono::duration<double>(end - start).count();
}

double dense_qr_from_scratch_time(const DenseMatrix& mat, const std::vector<double>& rhs, int base_rows, int update_row_index) {
    const int n = mat.cols;
    const int m_aug = base_rows + 1;
    const int lda_aug = m_aug;

    std::vector<double> A_aug(static_cast<std::size_t>(m_aug) * n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < base_rows; ++row) {
            A_aug[static_cast<std::size_t>(col) * lda_aug + row] = entry(mat, row, col);
        }
        A_aug[static_cast<std::size_t>(col) * lda_aug + base_rows] = entry(mat, update_row_index, col);
    }

    std::vector<double> b_aug(static_cast<std::size_t>(m_aug));
    for (int row = 0; row < base_rows; ++row) {
        b_aug[static_cast<std::size_t>(row)] = rhs[static_cast<std::size_t>(row)];
    }
    b_aug[static_cast<std::size_t>(base_rows)] = rhs[static_cast<std::size_t>(update_row_index)];

    std::vector<double> tau(static_cast<std::size_t>(std::min(m_aug, n)));

    int info = 0;
    int lwork = -1;
    double work_query = 0.0;

    auto start = Clock::now();

    dgeqrf_(const_cast<int*>(&m_aug), const_cast<int*>(&n), A_aug.data(), const_cast<int*>(&lda_aug), tau.data(), &work_query, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dense scratch dgeqrf workspace query failed");
    }
    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeqrf_(const_cast<int*>(&m_aug), const_cast<int*>(&n), A_aug.data(), const_cast<int*>(&lda_aug), tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dense scratch dgeqrf failed");
    }

    int nrhs = 1;
    int ldc = m_aug;
    lwork = -1;
    work_query = 0.0;
    char side = 'L';
    char trans = 'T';
    dormqr_(&side, &trans, const_cast<int*>(&m_aug), &nrhs, const_cast<int*>(&n), A_aug.data(), const_cast<int*>(&lda_aug), tau.data(), b_aug.data(), &ldc, &work_query, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dense scratch dormqr workspace query failed");
    }
    lwork = std::max(1, static_cast<int>(work_query));
    work.assign(static_cast<std::size_t>(lwork), 0.0);
    dormqr_(&side, &trans, const_cast<int*>(&m_aug), &nrhs, const_cast<int*>(&n), A_aug.data(), const_cast<int*>(&lda_aug), tau.data(), b_aug.data(), &ldc, work.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("dense scratch dormqr failed");
    }

    std::vector<double> R_aug(static_cast<std::size_t>(n) * n, 0.0);
    extract_R(A_aug, m_aug, n, R_aug);
    std::vector<double> c_first_n(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        c_first_n[static_cast<std::size_t>(i)] = b_aug[static_cast<std::size_t>(i)];
    }
    std::vector<double> solution;
    back_substitute_upper(n, R_aug, c_first_n, solution);

    auto end = Clock::now();
    (void)solution;
    return std::chrono::duration<double>(end - start).count();
}

cholmod_sparse* build_cholmod_matrix(const DenseMatrix& mat, int base_rows, int update_row_index, cholmod_common* common) {
    const int n = mat.cols;
    const int m_aug = base_rows + 1;

    const SuiteSparse_long max_nnz = static_cast<SuiteSparse_long>((static_cast<long long>(m_aug) * n));
    cholmod_triplet* trip = cholmod_l_allocate_triplet(m_aug, n, max_nnz, 0, CHOLMOD_REAL, common);
    if (!trip) {
        throw std::runtime_error("Failed to allocate cholmod triplet");
    }

    SuiteSparse_long* Ti = static_cast<SuiteSparse_long*>(trip->i);
    SuiteSparse_long* Tj = static_cast<SuiteSparse_long*>(trip->j);
    double* Tx = static_cast<double*>(trip->x);

    SuiteSparse_long nnz = 0;
    const double eps = 1e-12;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < base_rows; ++row) {
            double val = entry(mat, row, col);
            if (std::abs(val) > eps) {
                Ti[nnz] = row;
                Tj[nnz] = col;
                Tx[nnz] = val;
                ++nnz;
            }
        }
        double val_extra = entry(mat, update_row_index, col);
        if (std::abs(val_extra) > eps) {
            Ti[nnz] = base_rows;
            Tj[nnz] = col;
            Tx[nnz] = val_extra;
            ++nnz;
        }
    }
    trip->nnz = nnz;

    cholmod_sparse* A = cholmod_l_triplet_to_sparse(trip, nnz, common);
    cholmod_l_free_triplet(&trip, common);
    if (!A) {
        throw std::runtime_error("Failed to convert triplet to sparse matrix");
    }
    return A;
}

double sparse_qr_time(const DenseMatrix& mat, const std::vector<double>& rhs, int base_rows, int update_row_index, cholmod_common* common) {
    const int m_aug = base_rows + 1;

    cholmod_sparse* A = build_cholmod_matrix(mat, base_rows, update_row_index, common);

    cholmod_dense* b = cholmod_l_allocate_dense(m_aug, 1, m_aug, CHOLMOD_REAL, common);
    if (!b) {
        cholmod_l_free_sparse(&A, common);
        throw std::runtime_error("Failed to allocate cholmod dense vector");
    }
    double* bdata = static_cast<double*>(b->x);
    for (int row = 0; row < base_rows; ++row) {
        bdata[row] = rhs[static_cast<std::size_t>(row)];
    }
    bdata[base_rows] = rhs[static_cast<std::size_t>(update_row_index)];

    cholmod_sparse* Q = nullptr;
    cholmod_sparse* R = nullptr;
    SuiteSparse_long* E = nullptr;

    auto start = Clock::now();
    const SuiteSparse_long econ = mat.cols;
    const double tol = 1e-12;
    const int ordering = SPQR_ORDERING_DEFAULT;
    SuiteSparseQR_C_QR(ordering, tol, econ, A, &Q, &R, &E, common);
    auto end = Clock::now();

    cholmod_l_free_sparse(&Q, common);
    cholmod_l_free_sparse(&R, common);
    SuiteSparse_free(E);
    cholmod_l_free_sparse(&A, common);
    cholmod_l_free_dense(&b, common);

    return std::chrono::duration<double>(end - start).count();
}

struct GeneratorSpec {
    std::string name;
    Dataset (*builder)(int total_rows, int cols, std::mt19937_64& rng);
};

Dataset build_block_wrapper(int total_rows, int cols, std::mt19937_64& rng) {
    (void)rng;
    return make_block_diagonal_dataset(total_rows, cols);
}

Dataset build_circular_wrapper(int total_rows, int cols, std::mt19937_64& rng) {
    (void)rng;
    return make_circular_dataset(total_rows, cols);
}

} // namespace

int main() {
    cholmod_common common;
    cholmod_l_start(&common);

    const int updates = 50;
    const std::vector<int> base_sizes = {50, 100, 500, 1000, 2000};

    std::vector<GeneratorSpec> generators = {
        {"random_sparse", [](int total_rows, int cols, std::mt19937_64& rng) {
             return make_random_sparse_dataset(total_rows, cols, 0.02, rng);
         }},
        {"circular_graph", build_circular_wrapper},
        {"block_diagonal", build_block_wrapper},
    };

    std::mt19937_64 rng(2025);

    constexpr const char* kOutputPath = "../benchmarks/givens_benchmark_results.csv";
    std::ofstream out(kOutputPath);
    if (!out) {
        throw std::runtime_error("Failed to open output CSV file");
    }

    out << std::fixed << std::setprecision(9);
    out << "generator,mode,base_rows,num_cols,update_index,time_seconds\n";

    for (const auto& gen : generators) {
        for (int base_m : base_sizes) {
            const int n = std::max(2, base_m / 2);
            const int total_rows = base_m + updates;

            Dataset data = gen.builder(total_rows, n, rng);
            DenseBaseline baseline = factorize_dense_baseline(data.matrix, data.rhs, base_m);

            for (int upd = 0; upd < updates; ++upd) {
                const int row_idx = base_m + upd;

                double t_dense_givens = dense_givens_update_time(data.matrix, data.rhs, baseline, base_m, row_idx);
                double t_dense_scratch = dense_qr_from_scratch_time(data.matrix, data.rhs, base_m, row_idx);
                double t_sparse = sparse_qr_time(data.matrix, data.rhs, base_m, row_idx, &common);

                out << gen.name << ",dense_givens_update," << base_m << ',' << n << ',' << upd << ',' << t_dense_givens << '\n';
                out << gen.name << ",dense_scratch," << base_m << ',' << n << ',' << upd << ',' << t_dense_scratch << '\n';
                out << gen.name << ",sparse," << base_m << ',' << n << ',' << upd << ',' << t_sparse << '\n';
            }
        }
    }

    out.close();
    std::cout << "Wrote benchmark measurements to " << kOutputPath << '\n';
    cholmod_l_finish(&common);
    return 0;
}
