#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional> // For std::hash
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include "SuiteSparseQR_C.h"
#include "cholmod.h"

#include "hash_givens_conflict_analyzer.hpp"

extern "C" {
void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* a, int* lda, double* tau, double* c, int* ldc, double* work, int* lwork, int* info);
}

namespace hash_givens {

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


inline void hash_combine(std::size_t& seed, double v) {
    std::hash<double> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::size_t hash_constraint_row(const DenseMatrix& mat, int row, double rhs_val) {
    const int n = mat.cols;
    const int m = mat.rows;
    double norm_val = 0.0;
    const double eps = 1e-15;

    // Find the first non-zero element (or max-abs) to use for normalization
    for (int col = 0; col < n; ++col) {
        double val = entry(mat, row, col);
        if (std::abs(val) > std::abs(norm_val)) {
            norm_val = val;
        }
    }
    if (std::abs(norm_val) < eps && std::abs(rhs_val) > eps) {
        norm_val = rhs_val;
    }

    // If still zero, it's a zero row.
    if (std::abs(norm_val) < eps) {
        return 0;
    }

    std::size_t seed = 0;
    for (int col = 0; col < n; ++col) {
        hash_combine(seed, entry(mat, row, col) / norm_val);
    }
    hash_combine(seed, rhs_val / norm_val);
    return seed;
}

// find hash 
std::unordered_map <int, std::size_t> compute_hashes(
    const Dataset& data,
    const std::unordered_set<int>& active_rows) {
    std::unordered_map<int, std::size_t> hashes;
    for (int row_idx : active_rows) { // maybe parallelize? 
        hashes[row_idx] = hash_constraint_row(data.matrix, row_idx, data.rhs[row_idx]);
    }
    return hashes;
}

DetectedChange detect_change(
    const std::unordered_map<int, std::size_t>& old_hashes,
    const std::unordered_map<int, std::size_t>& new_hashes) {

    std::vector<int> added_rows;
    std::vector<int> deleted_rows;


    for (const auto& pair : old_hashes) {
        if (new_hashes.find(pair.first) == new_hashes.end()) {
            deleted_rows.push_back(pair.first);
        }
    }


    for (const auto& pair : new_hashes) {
        if (old_hashes.find(pair.first) == old_hashes.end()) {
            added_rows.push_back(pair.first);
        }
    }

    // find modified row 
    for (const auto& pair : new_hashes) {
        auto old_it = old_hashes.find(pair.first);
        if (old_it != old_hashes.end() && old_it->second != pair.second) {
            // This row was modified
            return {DetectedChangeType::Modify, pair.first};
        }
    }

    if (added_rows.size() == 1 && deleted_rows.empty()) {
        return {DetectedChangeType::Add, added_rows[0]};
    }
    if (deleted_rows.size() == 1 && added_rows.empty()) {
        return {DetectedChangeType::Delete, deleted_rows[0]};
    }
    if (added_rows.empty() && deleted_rows.empty()) {
        return {DetectedChangeType::None, -1};
    }

    // reset 
    return {DetectedChangeType::Reset, -1};
}

const char* change_type_name(DetectedChangeType type) {
    switch (type) {
        case DetectedChangeType::None:
            return "none";
        case DetectedChangeType::Add:
            return "add";
        case DetectedChangeType::Delete:
            return "delete";
        case DetectedChangeType::Modify:
            return "modify";
        case DetectedChangeType::Reset:
            return "reset";
        default:
            return "unknown";
    }
}

DetectedChangeType expected_change_for(UpdateOperation op) {
    switch (op) {
        case UpdateOperation::Add:
            return DetectedChangeType::Add;
        case UpdateOperation::Delete:
            return DetectedChangeType::Delete;
        case UpdateOperation::Modify:
            return DetectedChangeType::Modify;
        default:
            return DetectedChangeType::None;
    }
}

double compute_residual(
    const DenseMatrix& mat,
    const std::vector<double>& rhs,
    const std::unordered_set<int>& active_rows,
    const std::vector<double>& x) {
    if (active_rows.empty() || x.empty()) {
        return 0.0;
    }

    double sum_sq = 0.0;
    for (int row_idx : active_rows) {
        double ax = 0.0;
        for (int col = 0; col < mat.cols; ++col) {
            ax += entry(mat, row_idx, col) * x[static_cast<std::size_t>(col)];
        }
        double diff = ax - rhs[static_cast<std::size_t>(row_idx)];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}


// dense qr factoriziation 
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

// MOD: This function is now the "from_scratch" dense solver
// It takes a dynamic set of active rows
CachedDenseQR factorize_dense_from_scratch(
    const DenseMatrix& mat,
    const std::vector<double>& rhs,
    const std::unordered_set<int>& active_rows) {
    
    if (active_rows.empty()) {
        return {};
    }

    const int m = active_rows.size();
    const int n = mat.cols;
    const int lda = m;

    if (m < n) {
         // This benchmark assumes overdetermined, but handle this case
        std::cerr << "Warning: Underdetermined system (m < n) in factorize_dense_from_scratch." << std::endl;
    }

    std::vector<double> A_work(static_cast<std::size_t>(m) * n, 0.0);
    std::vector<double> b_work(static_cast<std::size_t>(m));
    
    int current_row = 0;
    for (int row_idx : active_rows) {
        for (int col = 0; col < n; ++col) {
            A_work[static_cast<std::size_t>(col) * lda + current_row] = entry(mat, row_idx, col);
        }
        b_work[static_cast<std::size_t>(current_row)] = rhs[static_cast<std::size_t>(row_idx)];
        current_row++;
    }

    std::vector<double> tau(static_cast<std::size_t>(std::min(m, n)));

    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgeqrf_(const_cast<int*>(&m), const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), &work_query, &lwork, &info);
    if (info != 0) throw std::runtime_error("dgeqrf workspace query failed");

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeqrf_(const_cast<int*>(&m), const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), work.data(), &lwork, &info);
    if (info != 0) throw std::runtime_error("dgeqrf factorization failed");

    CachedDenseQR result;
    result.n = n;
    extract_R(A_work, m, n, result.R);

    int nrhs = 1;
    int ldc = m;
    lwork = -1;
    work_query = 0.0;
    char side = 'L';
    char trans = 'T';
    dormqr_(&side, &trans, const_cast<int*>(&m), &nrhs, const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), b_work.data(), &ldc, &work_query, &lwork, &info);
    if (info != 0) throw std::runtime_error("dormqr workspace query failed");

    lwork = std::max(1, static_cast<int>(work_query));
    work.assign(static_cast<std::size_t>(lwork), 0.0);
    dormqr_(&side, &trans, const_cast<int*>(&m), &nrhs, const_cast<int*>(&n), A_work.data(), const_cast<int*>(&lda), tau.data(), b_work.data(), &ldc, work.data(), &lwork, &info);
    if (info != 0) throw std::runtime_error("dormqr apply failed");

    result.c.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        result.c[static_cast<std::size_t>(i)] = (i < m) ? b_work[static_cast<std::size_t>(i)] : 0.0;
    }
    result.active_rows = active_rows;
    
    return result;
}

// givens + hash function 
SolveStats time_cached_qr_update(
    const Dataset& data,
    const std::unordered_set<int>& new_active_rows,
    CachedDenseQR& cache, // MOD: Pass cache by reference to update it
    const DetectedChange& change,
    std::vector<double>& solution_out) {
    
    auto start = Clock::now();

    const int n = data.matrix.cols;
    std::vector<double> solution;
    
    if (change.type == DetectedChangeType::Add) {
        std::vector<double> R = cache.R;
        std::vector<double> c = cache.c;
        std::vector<double> new_row(static_cast<std::size_t>(n));
        for (int col = 0; col < n; ++col) {
            new_row[static_cast<std::size_t>(col)] = entry(data.matrix, change.rowIndex, col);
        }
        double rhs_extra = data.rhs[static_cast<std::size_t>(change.rowIndex)];

        for (int j = 0; j < n; ++j) {
            std::size_t diag_idx = static_cast<std::size_t>(j) * n + j;
            double diag = R[diag_idx];
            double w = new_row[static_cast<std::size_t>(j)];
            double r = std::hypot(diag, w);
            if (r == 0.0) continue;
            
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
        
        // Update the cache
        cache.R = std::move(R);
        cache.c = std::move(c);
        cache.active_rows = new_active_rows;
        
        back_substitute_upper(n, cache.R, cache.c, solution);

    } else {
        cache = factorize_dense_from_scratch(data.matrix, data.rhs, new_active_rows);
        if (cache.n > 0) {
            back_substitute_upper(n, cache.R, cache.c, solution);
        }
    }

    auto end = Clock::now();
    solution_out = std::move(solution);
    double residual = compute_residual(data.matrix, data.rhs, new_active_rows, solution_out);
    return {std::chrono::duration<double>(end - start).count(), residual};
}

// =========================================================================
// Benchmark 2: Dense QR from Scratch
// =========================================================================
SolveStats time_dense_qr_from_scratch(
    const Dataset& data,
    const std::unordered_set<int>& active_rows,
    std::vector<double>& solution_out) {
    
    auto start = Clock::now();
    
    CachedDenseQR result = factorize_dense_from_scratch(data.matrix, data.rhs, active_rows);
    std::vector<double> solution;
    if (result.n > 0) {
         back_substitute_upper(result.n, result.R, result.c, solution);
    }
    auto end = Clock::now();

    solution_out = std::move(solution);
    double residual = compute_residual(data.matrix, data.rhs, active_rows, solution_out);
    return {std::chrono::duration<double>(end - start).count(), residual};
}


// =========================================================================
// Benchmark 3: Sparse QR from Scratch
// =========================================================================
cholmod_sparse* build_cholmod_matrix(
    const DenseMatrix& mat,
    const std::vector<int>& ordered_rows,
    cholmod_common* common
) {
    const int n = mat.cols;
    const int m_aug = static_cast<int>(ordered_rows.size());

    const SuiteSparse_long max_nnz = static_cast<SuiteSparse_long>((static_cast<long long>(m_aug) * n));
    cholmod_triplet* trip = cholmod_l_allocate_triplet(m_aug, n, max_nnz, 0, CHOLMOD_REAL, common);
    if (!trip) throw std::runtime_error("Failed to allocate cholmod triplet");

    SuiteSparse_long* Ti = static_cast<SuiteSparse_long*>(trip->i);
    SuiteSparse_long* Tj = static_cast<SuiteSparse_long*>(trip->j);
    double* Tx = static_cast<double*>(trip->x);

    SuiteSparse_long nnz = 0;
    const double eps = 1e-12;
    for (int aug_row = 0; aug_row < m_aug; ++aug_row) {
        int orig_row = ordered_rows[static_cast<std::size_t>(aug_row)];
        for (int col = 0; col < n; ++col) {
            double val = entry(mat, orig_row, col);
            if (std::abs(val) > eps) {
                Ti[nnz] = aug_row;
                Tj[nnz] = col;
                Tx[nnz] = val;
                ++nnz;
            }
        }
    }
    trip->nnz = nnz;

    cholmod_sparse* A = cholmod_l_triplet_to_sparse(trip, nnz, common);
    cholmod_l_free_triplet(&trip, common);
    if (!A) throw std::runtime_error("Failed to convert triplet to sparse matrix");
    return A;
}

SolveStats time_sparse_qr_from_scratch(
    const Dataset& data,
    const std::vector<int>& ordered_rows,
    std::vector<double>& solution_out,
    cholmod_common* common) {

    if (ordered_rows.empty()) {
        solution_out.clear();
        return {0.0, 0.0};
    }

    const int m_aug = static_cast<int>(ordered_rows.size());
    cholmod_sparse* A = build_cholmod_matrix(data.matrix, ordered_rows, common);

    cholmod_dense* b = cholmod_l_allocate_dense(m_aug, 1, m_aug, CHOLMOD_REAL, common);
    if (!b) {
        cholmod_l_free_sparse(&A, common);
        throw std::runtime_error("Failed to allocate cholmod dense vector");
    }
    double* bdata = static_cast<double*>(b->x);
    for (int i = 0; i < m_aug; ++i) {
        bdata[i] = data.rhs[static_cast<std::size_t>(ordered_rows[static_cast<std::size_t>(i)])];
    }

    const double tol = 1e-12;
    const int ordering = SPQR_ORDERING_DEFAULT;

    auto start = Clock::now();
    cholmod_dense* x = SuiteSparseQR_C_backslash(ordering, tol, A, b, common);
    auto end = Clock::now();

    if (!x) {
        cholmod_l_free_sparse(&A, common);
        cholmod_l_free_dense(&b, common);
        throw std::runtime_error("SuiteSparseQR backslash solve failed");
    }

    double* xdata = static_cast<double*>(x->x);
    const int n = data.matrix.cols;
    solution_out.assign(static_cast<std::size_t>(n), 0.0);
    for (int col = 0; col < n; ++col) {
        solution_out[static_cast<std::size_t>(col)] = xdata[col];
    }

    cholmod_l_free_dense(&x, common);
    cholmod_l_free_sparse(&A, common);
    cholmod_l_free_dense(&b, common);

    std::unordered_set<int> active_rows_set(ordered_rows.begin(), ordered_rows.end());
    double residual = compute_residual(data.matrix, data.rhs, active_rows_set, solution_out);
    return {std::chrono::duration<double>(end - start).count(), residual};
}

// =========================================================================
// Benchmark 4: Kaczmarz Iterative Method (with Warm Start)
// =========================================================================
SolveStats time_kaczmarz_solve(
    const Dataset& data,
    const std::unordered_set<int>& active_rows_set,
    std::vector<double>& x_solution, 
    std::mt19937_64& rng
) {
    if (active_rows_set.empty()) {
        x_solution.clear();
        return {0.0, 0.0};
    }
    
    const int n = data.matrix.cols;
    if (x_solution.size() != static_cast<std::size_t>(n)) {
        x_solution.assign(n, 0.0);
    }
    
    // Convert set to vector for random indexing
    std::vector<int> active_rows(active_rows_set.begin(), active_rows_set.end());
    const int m = active_rows.size();
    
    std::vector<double> row_norms_sq(m, 0.0);
    std::vector<int> row_indices(m);
    for(int i = 0; i < m; ++i) {
        const int row_idx = active_rows[i];
        row_indices[i] = row_idx;
        double norm_sq = 0.0;
        for (int col = 0; col < n; ++col) {
            double val = entry(data.matrix, row_idx, col);
            norm_sq += val * val;
        }
        row_norms_sq[i] = (norm_sq == 0.0) ? 1.0 : norm_sq; // Avoid div by zero
    }
    
    std::uniform_int_distribution<int> dist(0, m - 1);
    
    // MOD: Run a fixed number of iterations (e.g., 3-5 passes)
    const int num_passes = 3;
    const int max_iters = num_passes * m;

    auto start = Clock::now();

    for (int iter = 0; iter < max_iters; ++iter) {
        const int i = dist(rng); // Randomized row index
        const int row_idx = row_indices[i];
        
        double dot = 0.0;
        for (int col = 0; col < n; ++col) {
            dot += entry(data.matrix, row_idx, col) * x_solution[col];
        }
        
        const double rhs_val = data.rhs[row_idx];
        const double alpha = (rhs_val - dot) / row_norms_sq[i];
        
        // Project solution onto the hyperplane
        for (int col = 0; col < n; ++col) {
            x_solution[col] += alpha * entry(data.matrix, row_idx, col);
        }
    }

    auto end = Clock::now();
    double residual = compute_residual(data.matrix, data.rhs, active_rows_set, x_solution);
    return {std::chrono::duration<double>(end - start).count(), residual};
}


// =lia======================================================================
// Dataset Builders
// =========================================================================

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

} // namespace hash_givens

int main() {
    cholmod_common common;
    cholmod_l_start(&common);

    const std::vector<int> sizes = {50, 100, 500, 1000, 2000, 5000, 10000};
    const int updates = 50;
    const double conflict_tol = 1e-6;
    const std::string output_path = "../benchmarks/hash_vs_sparse_comparison.csv";

    hash_givens::run_cached_vs_sparse_comparison(&common, sizes, updates, conflict_tol, output_path);
    cholmod_l_finish(&common);
    return 0;
}