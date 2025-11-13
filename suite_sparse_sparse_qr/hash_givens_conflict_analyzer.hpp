#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "SuiteSparseQR_C.h"
#include "cholmod.h"

namespace hash_givens {

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

struct CachedDenseQR {
    int n = 0;
    std::vector<double> R; // n x n upper-triangular
    std::vector<double> c; // first n entries of Q^T b
    std::unordered_set<int> active_rows;
};

enum class UpdateOperation {
    Add,
    Delete,
    Modify
};

enum class DetectedChangeType {
    None,
    Add,
    Delete,
    Modify,
    Reset
};

struct DetectedChange {
    DetectedChangeType type = DetectedChangeType::None;
    int rowIndex = -1;
};

struct SolveStats {
    double seconds = 0.0;
    double residual = 0.0;
};

inline double& entry(DenseMatrix& mat, int row, int col) {
    return mat.values[static_cast<std::size_t>(col) * mat.rows + row];
}

inline double entry(const DenseMatrix& mat, int row, int col) {
    return mat.values[static_cast<std::size_t>(col) * mat.rows + row];
}

std::vector<double> make_random_vector(int n, std::mt19937_64& rng);
Dataset make_random_sparse_dataset(int total_rows, int n, double density, std::mt19937_64& rng);
Dataset make_circular_dataset(int total_rows, int n);
Dataset make_block_diagonal_dataset(int total_rows, int n);

std::unordered_map<int, std::size_t> compute_hashes(
    const Dataset& data,
    const std::unordered_set<int>& active_rows);

DetectedChange detect_change(
    const std::unordered_map<int, std::size_t>& old_hashes,
    const std::unordered_map<int, std::size_t>& new_hashes);

DetectedChangeType expected_change_for(UpdateOperation op);

double compute_residual(
    const DenseMatrix& mat,
    const std::vector<double>& rhs,
    const std::unordered_set<int>& active_rows,
    const std::vector<double>& x);

void extract_R(const std::vector<double>& A_factored, int m, int n, std::vector<double>& R_out);

void back_substitute_upper(int n, const std::vector<double>& R, const std::vector<double>& c, std::vector<double>& x);

CachedDenseQR factorize_dense_from_scratch(
    const DenseMatrix& mat,
    const std::vector<double>& rhs,
    const std::unordered_set<int>& active_rows);

SolveStats time_cached_qr_update(
    const Dataset& data,
    const std::unordered_set<int>& new_active_rows,
    CachedDenseQR& cache,
    const DetectedChange& change,
    std::vector<double>& solution_out);

SolveStats time_dense_qr_from_scratch(
    const Dataset& data,
    const std::unordered_set<int>& active_rows,
    std::vector<double>& solution_out);

SolveStats time_sparse_qr_from_scratch(
    const Dataset& data,
    const std::vector<int>& ordered_rows,
    std::vector<double>& solution_out,
    cholmod_common* common);

SolveStats time_kaczmarz_solve(
    const Dataset& data,
    const std::unordered_set<int>& active_rows_set,
    std::vector<double>& x_solution,
    std::mt19937_64& rng);

struct ProbabilityScenario {
    std::string name;
    double p_add;
    double p_modify;
    double p_delete;
};

struct AggregatedRecord {
    int base_rows = 0;
    int num_cols = 0;
    std::string scenario;
    std::string solver;
    double mean_seconds = 0.0;
    double mean_residual = 0.0;
    double conflict_rate = 0.0;
    double correct_rate = 0.0;
    double detection_rate = 0.0;
};

std::vector<int> make_shuffled_rows(const std::unordered_set<int>& active_rows, std::mt19937_64& rng);

UpdateOperation pick_operation(
    const ProbabilityScenario& scenario,
    const std::unordered_set<int>& active_rows,
    const std::vector<int>& available_rows,
    int min_active_rows,
    std::mt19937_64& rng);

AggregatedRecord make_record(
    int base_rows,
    int num_cols,
    const ProbabilityScenario& scenario,
    const std::string& solver,
    double total_seconds,
    double total_residual,
    int conflict_count,
    int correct_count,
    int detection_count,
    int updates);

void write_records(const std::string& path, const std::vector<AggregatedRecord>& records);

void run_cached_vs_sparse_comparison(
    cholmod_common* common,
    const std::vector<int>& sizes,
    int updates,
    double conflict_tol,
    const std::string& output_path);

} // namespace hash_givens
