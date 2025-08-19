#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
#include <set>

#include "PiafMunch/PM_arrays.h"

// y = A(m x n, stored in column-major) * x(n)
// A is flattened as [col0 (m entries), col1 (m entries), ..., col{n-1}]
static std::vector<double> matvec_colmajor(const std::vector<double>& A,
                                           int m, int n,
                                           const std::vector<double>& x) {
    
    // Iterate columns (because A is column-major)
    assert((int)A.size() == m * n);
    assert((int)x.size() == n);
    std::vector<double> y(m, 0.0);
    for (int j = 0; j < n; ++j) {
        const double xj = x[j];
        const int base = j * m; // start of column j in column-major layout
        for (int i = 0; i < m; ++i) {
            y[i] += A[base + i] * xj;
        }
    }
    return y;
}

// Compare two vectors with a small absolute tolerance
static bool nearly_equal(const std::vector<double>& a,
                         const std::vector<double>& b,
                         double eps = 1e-12) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) return false;
    }
    return true;
}

// Helper to build a {-1,0,+1} sparse matrix by explicit (i,j,val) triplets.
// NOTE: indices are 1-based to match the library conventions.
static Sparse_matrix make_sparse_unit_from_triplets(
    int m, int n,
    const std::vector<std::tuple<int,int,int>>& triplets // (i,j,val in {-1,+1})
) {
    // Pre-allocate with the exact nnz capacity to avoid reallocations.
    // This constructor exists in the library: Sparse_matrix(m, n, npnz)
    Sparse_matrix S(m, n, (int)triplets.size());

    // Set each (i,j) entry to +/-1 using add_(..., ad=0) which means "set".
    for (auto [i, j, v] : triplets) {
        assert(v == 1 || v == -1);
        S.add_(i, j, (double)v, 0);  // set S(i,j) = v
    }
    return S;
}

// Single test driver: build U, multiply both paths, compare.
static void run_case_general(int m, int n,
                             const std::vector<std::tuple<int,int,int>>& triplets,
                             const std::vector<double>& x_cpp,
                             const char* name) {
    // 1. Build Sparse_matrix with explicit +-1 entries
    Sparse_matrix S = make_sparse_unit_from_triplets(m, n, triplets);

    // 2. Convert to SpUnit_matrix (requires entries in {-1,0,+1})
    SpUnit_matrix U(S);

    // 3. Make Fortran_vector x from std::vector
    Fortran_vector x(x_cpp);

    // 4. Reference product via library path: y_ref = U * x
    Fortran_vector y_ref(m, 0.0);
    y_ref.set_matmult(U, x);

    // 5. Dense path: U -> dense col-major -> y_dense = A * x
    std::vector<double> A_colmajor = U.toCppVector();
    std::vector<double> y_dense = matvec_colmajor(A_colmajor, m, n, x_cpp);

    // 6. Compare
    auto y_ref_cpp = y_ref.toCppVector();
    if (!nearly_equal(y_dense, y_ref_cpp)) {
        std::cerr << "[FAIL] " << name << "\n";
        std::cerr << "y_dense: ";
        for (double v : y_dense) std::cerr << v << " ";
        std::cerr << "\n";
        std::cerr << "y_ref  : ";
        for (double v : y_ref_cpp) std::cerr << v << " ";
        std::cerr << "\n";
        std::exit(1);
    } else {
        std::cout << "[OK] " << name << "\n";
    }
}

int main() {
    // identity 3x3
    {
        int m=3, n=3;
        std::vector<std::tuple<int,int,int>> T = {{1,1, +1},{2,2, +1},{3,3, +1}};
        run_case_general(m, n, T, {1,2,3}, "identity 3x3");
    }

    // negative diagonal 4x4
    {
        int m=4, n=4;
        std::vector<std::tuple<int,int,int>> T = {{1,1, -1},{2,2, -1},{3,3, -1},{4,4, -1}};
        run_case_general(m, n, T, {1,2,3,4}, "neg diag 4x4");
    }

    // rectangular 3x4: ones on the first 3 diagonal positions, last col zeros
    {
        int m=3, n=4;
        std::vector<std::tuple<int,int,int>> T = {{1,1, +1},{2,2, +1},{3,3, +1}};
        run_case_general(m, n, T, {10,20,30,40}, "rectangular 3x4 diag");
    }

    // permutation matrix 5x5 (cyclic shift)
    // P maps e1->e2, e2->e3, e3->e4, e4->e5, e5->e1
    {
        int m=5, n=5;
        std::vector<std::tuple<int,int,int>> T = {
            {2,1,+1}, {3,2,+1}, {4,3,+1}, {5,4,+1}, {1,5,+1}
        };
        run_case_general(m, n, T, {1,2,3,4,5}, "permutation (cyclic shift) 5x5");
    }

    // upper bidiagonal 4x4 with +1 on diag and -1 on superdiag
    {
        int m=4, n=4;
        std::vector<std::tuple<int,int,int>> T = {
            {1,1,+1}, {2,2,+1}, {3,3,+1}, {4,4,+1},
            {1,2,-1}, {2,3,-1}, {3,4,-1}
        };
        run_case_general(m, n, T, {2,0,-1,3}, "upper bidiagonal (+I, -superdiag) 4x4");
    }

    // random sparse +/-1 pattern (repeatable)
    {
        int m=6, n=5, nnz=8;
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> r_i(1, m), r_j(1, n), r_s(0,1);

        std::vector<std::tuple<int,int,int>> T;
        T.reserve(nnz);
        // Generate distinct positions (simple retry scheme)
        std::set<std::pair<int,int>> used;
        while ((int)T.size() < nnz) {
            int i = r_i(rng), j = r_j(rng);
            if (!used.insert({i,j}).second) continue;
            int s = r_s(rng) ? +1 : -1;
            T.emplace_back(i, j, s);
        }
        run_case_general(m, n, T, {1,2,3,4,5}, "random sparse +/-1 (seed=12345)");
    }

    std::cout << "All extended tests passed.\n";
    return 0;
}