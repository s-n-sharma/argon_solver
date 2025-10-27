#include <iostream>
#include <stdexcept>

#include "SuiteSparseQR_C.h" // Main SPQR header
#include "cholmod.h"         // Header for sparse matrix data structures

#include "suite_sparse_sparse_qr/qr_conflict_analyzer.hpp"

int main() {
    // --- 1. Start CHOLMOD ---
    cholmod_common c;
    cholmod_l_start(&c);

    // --- 2. Define the sparse matrix A (4x3) ---
    // (row, col, value)
    cholmod_triplet *T = cholmod_l_allocate_triplet(4, 3, 6, 0, CHOLMOD_REAL, &c);
    if (T == NULL) {
        std::cerr << "Failed to allocate triplet." << std::endl;
        cholmod_l_finish(&c);
        return 1;
    }

    long *Ti = (long*)T->i; // row indices
    long *Tj = (long*)T->j; // col indices
    double *Tx = (double*)T->x; // values

    Ti[0] = 0; Tj[0] = 0; Tx[0] = 10.0; // A[0,0]
    Ti[1] = 1; Tj[1] = 1; Tx[1] = 20.0; // A[1,1]
    Ti[2] = 2; Tj[2] = 2; Tx[2] = 30.0; // A[2,2]
    Ti[3] = 3; Tj[3] = 0; Tx[3] = 1.0;  // A[3,0]
    Ti[4] = 3; Tj[4] = 1; Tx[4] = 1.0;  // A[3,1]
    Ti[5] = 3; Tj[5] = 2; Tx[5] = 1.0;  // A[3,2]
    T->nnz = 6;

    cholmod_sparse *A = cholmod_l_triplet_to_sparse(T, T->nnz, &c);
    cholmod_l_free_triplet(&T, &c); // Free the triplet form

    // --- 3. Define the dense right-hand-side b (4x1) ---
    cholmod_dense *b = cholmod_l_allocate_dense(4, 1, 4, CHOLMOD_REAL, &c);
    double *bx = (double*)b->x;
    bx[0] = 1.0;
    bx[1] = 2.0;
    bx[2] = 3.0;
    bx[3] = 4.0;

    // --- 4. Solve the least-squares problem: min ||Ax - b|| ---
    cholmod_dense *x = SuiteSparseQR_C_backslash(
        SPQR_ORDERING_DEFAULT,
        SPQR_DEFAULT_TOL,
        A,
        b,
        &c
    );

    // --- 5. Print the solution x (3x1) ---
    if (x != NULL) {
        double *xx = (double*)x->x;
        std::cout << "Solution x:" << std::endl;
        for (size_t i = 0; i < x->nrow; ++i) {
            std::cout << "x[" << i << "] = " << xx[i] << std::endl;
        }
    } else {
        std::cerr << "Solver failed." << std::endl;
    }

    // --- 6. Run conflict analysis using rank-revealing QR ---
    try {
        QRConflictAnalysisResult analysis = analyze_system_with_qr(
            A,
            b,
            1e-10,
            1e-9,
            &c,
            true
        );

        std::cout << "\n--- Conflict Analysis ---" << std::endl;
        std::cout << "Rank estimate: " << analysis.rank << std::endl;
        std::cout << "Is underconstrained: " << std::boolalpha << analysis.isUnderconstrained << std::endl;
        std::cout << "Is conflicting: " << std::boolalpha << analysis.isConflicting << std::endl;
        std::cout << "Conflict norm: " << analysis.conflictNorm << std::endl;

        std::cout << "Residuals (row index : value):" << std::endl;
        for (SuiteSparse_long idx : analysis.sortedIndices) {
            std::cout << "  " << idx << " : "
                      << analysis.residual[static_cast<std::size_t>(idx)]
                      << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Conflict analysis failed: " << ex.what() << std::endl;
    }

    // --- 7. Clean up memory ---
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_free_dense(&b, &c);
    cholmod_l_free_dense(&x, &c);
    cholmod_l_finish(&c);

    return 0;
}
