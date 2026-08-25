#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "sankhya/lu.hpp"

using sankhya::Int;
using sankhya::LuFactor;
using sankhya::LuOptions;
using sankhya::SparseMatrix;
using sankhya::sz;
using sankhya::Triplet;

namespace {

// A wrong factorisation is the worst defect available in this project: every
// simplex iteration would use it, and nothing downstream would report anything
// wrong - the answers would simply drift. So the property checked here is the
// only one that matters, and it is checked directly rather than inferred:
// solving with the factors must reproduce the right-hand side when multiplied
// back through the original matrix.

// The factoriser takes the matrix column-wise, meaning row i of `columns` holds
// column i of the basis.
SparseMatrix column_store(Int n, const std::vector<std::vector<double>>& dense) {
  std::vector<Triplet> t;
  for (Int j = 0; j < n; ++j) {
    for (Int i = 0; i < n; ++i) {
      if (dense[sz(i)][sz(j)] != 0.0) t.push_back(Triplet{j, i, dense[sz(i)][sz(j)]});
    }
  }
  return SparseMatrix::from_triplets(n, n, std::move(t));
}

void multiply_dense(const std::vector<std::vector<double>>& b,
                    const std::vector<double>& x, std::vector<double>* y) {
  const std::size_t n = x.size();
  y->assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) sum += b[i][j] * x[j];
    (*y)[i] = sum;
  }
}

void multiply_transpose_dense(const std::vector<std::vector<double>>& b,
                              const std::vector<double>& x,
                              std::vector<double>* y) {
  const std::size_t n = x.size();
  y->assign(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += b[i][j] * x[i];
    (*y)[j] = sum;
  }
}

// Factorise, solve, multiply back, and report the worst discrepancy.
double round_trip_error(const std::vector<std::vector<double>>& dense,
                        const std::vector<double>& rhs, bool transpose,
                        const LuFactor** out = nullptr) {
  const Int n = static_cast<Int>(dense.size());
  static LuFactor lu;
  lu = LuFactor();
  std::vector<Int> basis(sz(n));
  for (Int j = 0; j < n; ++j) basis[sz(j)] = j;
  std::string error;
  if (!lu.factorize(column_store(n, dense), basis, LuOptions{}, &error)) {
    std::fprintf(stderr, "factorize failed: %s\n", error.c_str());
    return 1e9;
  }
  if (out) *out = &lu;

  std::vector<double> x = rhs;
  if (transpose) {
    lu.btran(&x);
  } else {
    lu.ftran(&x);
  }
  std::vector<double> back;
  if (transpose) {
    multiply_transpose_dense(dense, x, &back);
  } else {
    multiply_dense(dense, x, &back);
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    const double scale = std::fmax(1.0, std::fabs(rhs[i]));
    worst = std::fmax(worst, std::fabs(back[i] - rhs[i]) / scale);
  }
  return worst;
}

void test_identity() {
  const std::vector<std::vector<double>> b = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const std::vector<double> rhs = {3.0, -1.0, 7.5};
  CHECK(round_trip_error(b, rhs, false) < 1e-14);
  CHECK(round_trip_error(b, rhs, true) < 1e-14);
}

void test_permutation() {
  // Nothing to eliminate, but everything to permute. Gets the row and column
  // orderings wrong if they are confused with each other.
  const std::vector<std::vector<double>> b = {
      {0, 0, 2}, {3, 0, 0}, {0, -4, 0}};
  const std::vector<double> rhs = {1.0, 2.0, 3.0};
  const LuFactor* lu = nullptr;
  CHECK(round_trip_error(b, rhs, false, &lu) < 1e-13);
  CHECK(round_trip_error(b, rhs, true) < 1e-13);
  if (lu) CHECK_EQ(lu->nucleus_size(), 0);  // pure singleton peeling
}

void test_triangular_needs_no_elimination() {
  const std::vector<std::vector<double>> b = {
      {2, 0, 0, 0}, {1, 3, 0, 0}, {-1, 2, 4, 0}, {5, 0, 1, 2}};
  const std::vector<double> rhs = {1.0, -2.0, 3.0, 0.5};
  const LuFactor* lu = nullptr;
  CHECK(round_trip_error(b, rhs, false, &lu) < 1e-13);
  CHECK(round_trip_error(b, rhs, true) < 1e-13);
  if (lu) {
    CHECK_EQ(lu->nucleus_size(), 0);
    std::printf("     triangular 4x4: nucleus %d, fill ratio %.2f\n",
                lu->nucleus_size(), lu->fill_ratio());
  }
}

void test_needs_elimination() {
  // No singleton anywhere, so every pivot goes through Markowitz.
  const std::vector<std::vector<double>> b = {
      {4, 1, 2}, {1, 5, 3}, {2, 3, 6}};
  const std::vector<double> rhs = {1.0, 2.0, 3.0};
  const LuFactor* lu = nullptr;
  CHECK(round_trip_error(b, rhs, false, &lu) < 1e-12);
  CHECK(round_trip_error(b, rhs, true) < 1e-12);
  if (lu) CHECK(lu->nucleus_size() > 0);
}

void test_singular_is_detected() {
  // Third column is the sum of the first two.
  const std::vector<std::vector<double>> b = {
      {1, 0, 1}, {0, 1, 1}, {1, 1, 2}};
  LuFactor lu;
  std::vector<Int> basis = {0, 1, 2};
  std::string error;
  CHECK(!lu.factorize(column_store(3, b), basis, LuOptions{}, &error));
  CHECK(!error.empty());
}

void test_ill_conditioned_still_round_trips() {
  // Entries spanning eight orders of magnitude. The threshold rule should keep
  // this stable enough to solve, which is the whole reason it exists.
  const std::vector<std::vector<double>> b = {
      {1e-4, 1.0, 0.0}, {1.0, 1e4, 1.0}, {0.0, 1.0, 1e-3}};
  const std::vector<double> rhs = {1.0, 1.0, 1.0};
  const double e = round_trip_error(b, rhs, false);
  std::printf("     ill-conditioned 3x3: round trip error %.2e\n", e);
  CHECK(e < 1e-8);
}

void test_random_sparse_matrices() {
  // The real test. Many random bases, each solved both ways and multiplied
  // back. A permutation or an index-space mistake shows up here immediately and
  // nowhere else.
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<double> value(-4.0, 4.0);
  double worst_ftran = 0.0;
  double worst_btran = 0.0;
  double worst_fill = 0.0;
  int factorized = 0;

  for (int trial = 0; trial < 300; ++trial) {
    const Int n = 4 + static_cast<Int>(rng() % 40);
    std::vector<std::vector<double>> b(sz(n), std::vector<double>(sz(n), 0.0));
    // A dominant diagonal keeps most draws nonsingular; off-diagonals are
    // sparse so the singleton phase has something to do.
    for (Int i = 0; i < n; ++i) {
      b[sz(i)][sz(i)] = 2.0 + std::fabs(value(rng));
      const int extra = static_cast<int>(rng() % 4);
      for (int k = 0; k < extra; ++k) {
        const Int j = static_cast<Int>(rng() % static_cast<unsigned>(n));
        if (j != i) b[sz(i)][sz(j)] = value(rng);
      }
    }
    std::vector<double> rhs(sz(n));
    for (double& r : rhs) r = value(rng);

    const LuFactor* lu = nullptr;
    const double ef = round_trip_error(b, rhs, false, &lu);
    const double eb = round_trip_error(b, rhs, true);
    if (ef > 1e8) continue;  // singular draw, skipped
    ++factorized;
    worst_ftran = std::fmax(worst_ftran, ef);
    worst_btran = std::fmax(worst_btran, eb);
    if (lu) worst_fill = std::fmax(worst_fill, lu->fill_ratio());
  }

  std::printf("     %d random bases: worst ftran %.2e, worst btran %.2e, "
              "worst fill ratio %.2f\n",
              factorized, worst_ftran, worst_btran, worst_fill);
  CHECK(factorized > 250);
  CHECK(worst_ftran < 1e-9);
  CHECK(worst_btran < 1e-9);
}

void test_fill_stays_reasonable() {
  // An arrowhead matrix: dense last row and column, sparse elsewhere. Pivoting
  // in the wrong order turns it completely dense, so the fill ratio is a direct
  // check that Markowitz is choosing sensibly.
  const Int n = 30;
  std::vector<std::vector<double>> b(sz(n), std::vector<double>(sz(n), 0.0));
  for (Int i = 0; i < n; ++i) b[sz(i)][sz(i)] = 3.0;
  for (Int i = 0; i < n - 1; ++i) {
    b[sz(i)][sz(n - 1)] = 1.0;
    b[sz(n - 1)][sz(i)] = 1.0;
  }
  std::vector<double> rhs(sz(n), 1.0);
  const LuFactor* lu = nullptr;
  const double e = round_trip_error(b, rhs, false, &lu);
  CHECK(e < 1e-11);
  if (lu) {
    std::printf("     arrowhead 30x30: fill ratio %.2f (dense would be about %.0f)\n",
                lu->fill_ratio(), static_cast<double>(n * n) / (3.0 * n));
    CHECK(lu->fill_ratio() < 2.0);
  }
}

}  // namespace

int main() {
  test_identity();
  test_permutation();
  test_triangular_needs_no_elimination();
  test_needs_elimination();
  test_singular_is_detected();
  test_ill_conditioned_still_round_trips();
  test_random_sparse_matrices();
  test_fill_stays_reasonable();
  return sankhya_test::finish("test_lu");
}
