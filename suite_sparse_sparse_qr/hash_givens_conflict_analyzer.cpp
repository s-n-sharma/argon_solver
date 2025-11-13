#include "hash_givens_conflict_analyzer.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace hash_givens {

std::vector<int> make_shuffled_rows(const std::unordered_set<int>& active_rows, std::mt19937_64& rng) {
    std::vector<int> rows(active_rows.begin(), active_rows.end());
    std::shuffle(rows.begin(), rows.end(), rng);
    return rows;
}

UpdateOperation pick_operation(
    const ProbabilityScenario& scenario,
    const std::unordered_set<int>& active_rows,
    const std::vector<int>& available_rows,
    int min_active_rows,
    std::mt19937_64& rng) {

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto can_add = [&]() { return !available_rows.empty(); };
    auto can_delete = [&]() { return static_cast<int>(active_rows.size()) > min_active_rows; };
    auto can_modify = [&]() { return !active_rows.empty(); };

    for (int attempt = 0; attempt < 10; ++attempt) {
        double r = dist(rng);
        UpdateOperation op;
        if (r < scenario.p_add) {
            op = UpdateOperation::Add;
        } else if (r < scenario.p_add + scenario.p_modify) {
            op = UpdateOperation::Modify;
        } else {
            op = UpdateOperation::Delete;
        }

        if (op == UpdateOperation::Add && can_add()) {
            return op;
        }
        if (op == UpdateOperation::Delete && can_delete()) {
            return op;
        }
        if (op == UpdateOperation::Modify && can_modify()) {
            return op;
        }
    }

    if (can_modify()) return UpdateOperation::Modify;
    if (can_add()) return UpdateOperation::Add;
    if (can_delete()) return UpdateOperation::Delete;
    return UpdateOperation::Modify;
}

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
    int updates) {

    AggregatedRecord rec;
    rec.base_rows = base_rows;
    rec.num_cols = num_cols;
    rec.scenario = scenario.name;
    rec.solver = solver;
    rec.mean_seconds = (updates > 0) ? total_seconds / updates : 0.0;
    rec.mean_residual = (updates > 0) ? total_residual / updates : 0.0;
    rec.conflict_rate = (updates > 0) ? static_cast<double>(conflict_count) / updates : 0.0;
    rec.correct_rate = (updates > 0) ? static_cast<double>(correct_count) / updates : 0.0;
    rec.detection_rate = (updates > 0) ? static_cast<double>(detection_count) / updates : 0.0;
    return rec;
}

void write_records(const std::string& path, const std::vector<AggregatedRecord>& records) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open comparison CSV file");
    }

    out << std::fixed << std::setprecision(9);
    out << "base_rows,num_cols,scenario,solver,mean_time_seconds,mean_residual,conflict_rate,correct_rate,detection_rate\n";
    for (const auto& rec : records) {
        out << rec.base_rows << ','
            << rec.num_cols << ','
            << rec.scenario << ','
            << rec.solver << ','
            << rec.mean_seconds << ','
            << rec.mean_residual << ','
            << rec.conflict_rate << ','
            << rec.correct_rate << ','
            << rec.detection_rate << '\n';
    }
}

void run_cached_vs_sparse_comparison(
    cholmod_common* common,
    const std::vector<int>& sizes,
    int updates,
    double conflict_tol,
    const std::string& output_path) {

    std::vector<ProbabilityScenario> scenarios = {
        {"equal_prob", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
        {"add_modify_heavy", 0.4, 0.4, 0.2},
    };

    std::mt19937_64 rng(1337);
    std::vector<AggregatedRecord> records;
    records.reserve(scenarios.size() * sizes.size() * 2);

    for (const auto& scenario : scenarios) {
        for (int base_m : sizes) {
            const int n = std::max(2, base_m / 2);
            const int extra_pool = std::max(200, updates * 2);
            const int total_rows = base_m + extra_pool;

            Dataset data = make_random_sparse_dataset(total_rows, n, 0.02, rng);

            std::unordered_set<int> active_rows;
            active_rows.reserve(static_cast<std::size_t>(base_m));
            for (int i = 0; i < base_m; ++i) {
                active_rows.insert(i);
            }

            std::vector<int> available_rows;
            available_rows.reserve(static_cast<std::size_t>(extra_pool));
            for (int i = base_m; i < total_rows; ++i) {
                available_rows.push_back(i);
            }

            CachedDenseQR cache = factorize_dense_from_scratch(data.matrix, data.rhs, active_rows);
            std::vector<double> cached_solution(static_cast<std::size_t>(n), 0.0);
            if (cache.n > 0) {
                back_substitute_upper(cache.n, cache.R, cache.c, cached_solution);
            }

            std::vector<double> sparse_solution(static_cast<std::size_t>(n), 0.0);

            auto current_hashes = compute_hashes(data, active_rows);

            double total_time_cached = 0.0;
            double total_res_cached = 0.0;
            int conflict_cached = 0;
            int correct_cached = 0;
            int detection_cached = 0;

            double total_time_sparse = 0.0;
            double total_res_sparse = 0.0;
            int conflict_sparse = 0;
            int correct_sparse = updates;

            for (int upd = 0; upd < updates; ++upd) {
                UpdateOperation op = pick_operation(scenario, active_rows, available_rows, n, rng);

                if (op == UpdateOperation::Add) {
                    std::uniform_int_distribution<int> dist(0, static_cast<int>(available_rows.size()) - 1);
                    int idx = dist(rng);
                    int row = available_rows[static_cast<std::size_t>(idx)];
                    active_rows.insert(row);
                    std::swap(available_rows[static_cast<std::size_t>(idx)], available_rows.back());
                    available_rows.pop_back();

                    if (!cached_solution.empty()) {
                        double rhs_val = 0.0;
                        for (int col = 0; col < n; ++col) {
                            rhs_val += entry(data.matrix, row, col) * cached_solution[static_cast<std::size_t>(col)];
                        }
                        data.rhs[static_cast<std::size_t>(row)] = rhs_val;
                    }

                } else if (op == UpdateOperation::Delete) {
                    std::uniform_int_distribution<int> dist(0, static_cast<int>(active_rows.size()) - 1);
                    auto it = active_rows.begin();
                    std::advance(it, dist(rng));
                    int row = *it;
                    active_rows.erase(it);
                    available_rows.push_back(row);

                } else { // Modify
                    std::uniform_int_distribution<int> dist(0, static_cast<int>(active_rows.size()) - 1);
                    auto it = active_rows.begin();
                    std::advance(it, dist(rng));
                    int row = *it;

                    std::normal_distribution<double> noise(0.0, 0.3);
                    bool changed = false;
                    for (int col = 0; col < n; ++col) {
                        double delta = noise(rng);
                        if (std::abs(delta) > 1e-12) {
                            entry(data.matrix, row, col) += delta;
                            changed = true;
                        }
                    }
                    if (!changed && n > 0) {
                        entry(data.matrix, row, 0) += 0.3;
                    }

                    std::normal_distribution<double> rhs_noise(0.0, 0.5);
                    data.rhs[static_cast<std::size_t>(row)] += rhs_noise(rng);
                }

                auto old_hashes = std::move(current_hashes);
                current_hashes = compute_hashes(data, active_rows);
                DetectedChange change = detect_change(old_hashes, current_hashes);
                DetectedChangeType expected = expected_change_for(op);
                if (change.type == expected) {
                    detection_cached += 1;
                }

                std::vector<int> ordered_rows = make_shuffled_rows(active_rows, rng);

                SolveStats stats_cached = time_cached_qr_update(data, active_rows, cache, change, cached_solution);
                SolveStats stats_sparse = time_sparse_qr_from_scratch(data, ordered_rows, sparse_solution, common);

                bool cached_conflict = stats_cached.residual > conflict_tol;
                bool sparse_conflict = stats_sparse.residual > conflict_tol;

                total_time_cached += stats_cached.seconds;
                total_res_cached += stats_cached.residual;
                conflict_cached += cached_conflict ? 1 : 0;
                correct_cached += (cached_conflict == sparse_conflict) ? 1 : 0;

                total_time_sparse += stats_sparse.seconds;
                total_res_sparse += stats_sparse.residual;
                conflict_sparse += sparse_conflict ? 1 : 0;
            }

            records.push_back(make_record(base_m, n, scenario, "cached_hash_qr", total_time_cached, total_res_cached, conflict_cached, correct_cached, detection_cached, updates));
            records.push_back(make_record(base_m, n, scenario, "sparse_qr", total_time_sparse, total_res_sparse, conflict_sparse, correct_sparse, 0, updates));
        }
    }

    write_records(output_path, records);
    std::cout << "Wrote cached-vs-sparse comparison to " << output_path << '\n';
}

} // namespace hash_givens
