// A factorisation either reproduces the matrix it factorised or it does not,
// so every test here builds M, picks x, forms b = M x, solves, and compares -
// against x itself and against a dense Cholesky written out in full below,
// which shares no code with the sparse one.
#include "anukulan/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"

using anukulan::CholeskyFactor;
using anukulan::CholeskyOptions;
using anukulan::Int;
using anukulan::SparseMatrix;
using anukulan::Triplet;
using anukulan::sz;

namespace {

using Dense = std::vector<std::vector<double>>;

Dense to_dense_symmetric(const SparseMatrix& lower) {
  const Int n = lower.rows();
  Dense d(sz(n), std::vector<double>(sz(n), 0.0));
  for (Int i = 0; i < n; ++i) {
    for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
      const Int j = lower.index()[sz(e)];
      if (j > i) continue;
      d[sz(i)][sz(j)] += lower.value()[sz(e)];
      if (j != i) d[sz(j)][sz(i)] += lower.value()[sz(e)];
    }
  }
  return d;
}

// Textbook dense Cholesky and solve, for reference. Returns false if the
// matrix is not numerically positive definite.
bool dense_solve(Dense m, std::vector<double>* b) {
  const std::size_t n = m.size();
  for (std::size_t j = 0; j < n; ++j) {
    double d = m[j][j];
    for (std::size_t k = 0; k < j; ++k) d -= m[j][k] * m[j][k];
    if (!(d > 0.0)) return false;
    m[j][j] = std::sqrt(d);
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = m[i][j];
      for (std::size_t k = 0; k < j; ++k) s -= m[i][k] * m[j][k];
      m[i][j] = s / m[j][j];
    }
  }
  std::vector<double>& x = *b;
  for (std::size_t i = 0; i < n; ++i) {
    double s = x[i];
    for (std::size_t k = 0; k < i; ++k) s -= m[i][k] * x[k];
    x[i] = s / m[i][i];
  }
  for (std::size_t i = n; i-- > 0;) {
    double s = x[i];
    for (std::size_t k = i + 1; k < n; ++k) s -= m[k][i] * x[k];
    x[i] = s / m[i][i];
  }
  return true;
}

std::vector<double> multiply(const Dense& m, const std::vector<double>& x) {
  std::vector<double> y(m.size(), 0.0);
  for (std::size_t i = 0; i < m.size(); ++i)
    for (std::size_t j = 0; j < m.size(); ++j) y[i] += m[i][j] * x[j];
  return y;
}

// The lower triangle of A D A' + shift I, which is the shape the interior
// point method hands over. A is rows x cols with `density` of its entries set.
SparseMatrix normal_matrix(std::mt19937* rng, Int rows, Int cols, double density,
                           double shift) {
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> weight(-6.0, 6.0);
  std::vector<std::vector<std::pair<Int, double>>> column(sz(cols));
  for (Int j = 0; j < cols; ++j) {
    for (Int i = 0; i < rows; ++i) {
      if (unit(*rng) < density) column[sz(j)].push_back({i, value(*rng)});
    }
  }
  std::vector<Triplet> entries;
  for (Int j = 0; j < cols; ++j) {
    // Weights spread over twelve orders of magnitude, like an interior point
    // iterate part of the way to an optimum.
    const double d = std::pow(10.0, weight(*rng));
    for (const auto& a : column[sz(j)]) {
      for (const auto& b : column[sz(j)]) {
        if (b.first > a.first) continue;
        entries.push_back({a.first, b.first, d * a.second * b.second});
      }
    }
  }
  for (Int i = 0; i < rows; ++i) entries.push_back({i, i, shift});
  return SparseMatrix::from_triplets(rows, rows, std::move(entries));
}

double worst_difference(const std::vector<double>& a, const std::vector<double>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double scale = std::fmax(1.0, std::fabs(b[i]));
    worst = std::fmax(worst, std::fabs(a[i] - b[i]) / scale);
  }
  return worst;
}

void test_random_spd_against_dense() {
  std::mt19937 rng(20260927);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
  double worst_sparse = 0.0;
  double worst_vs_dense = 0.0;
  int checked = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Int rows = 5 + static_cast<Int>(rng() % 60);
    const Int cols = rows + static_cast<Int>(rng() % 40);
    const double density = 0.03 + 0.2 * std::uniform_real_distribution<double>(0, 1)(rng);
    const SparseMatrix lower = normal_matrix(&rng, rows, cols, density, 1.0);
    const Dense dense = to_dense_symmetric(lower);

    std::vector<double> x_true(sz(rows));
    for (double& v : x_true) v = value(rng);
    const std::vector<double> b = multiply(dense, x_true);

    for (const bool natural : {false, true}) {
      CholeskyOptions options;
      if (natural) options.ordering = CholeskyOptions::Ordering::kNatural;
      if (trial % 3 == 0) options.relax_columns = 1;  // fundamental supernodes only
      CholeskyFactor factor;
      std::string error;
      CHECK(factor.analyse(lower, options, &error));
      CHECK(factor.factorize(lower, &error));
      CHECK_EQ(factor.dropped_pivots(), 0);
      std::vector<double> x = b;
      factor.solve(&x);
      std::vector<double> reference = b;
      CHECK(dense_solve(dense, &reference));
      worst_sparse = std::fmax(worst_sparse, worst_difference(x, x_true));
      worst_vs_dense = std::fmax(worst_vs_dense, worst_difference(x, reference));
      ++checked;
    }
  }
  // The weights span twelve orders of magnitude, so the matrices are badly
  // conditioned on purpose; what is compared is the sparse answer against the
  // dense one, which suffers the same conditioning.
  CHECK(worst_vs_dense < 1e-6);
  std::printf("     %d random normal-equation matrices: worst error vs dense %.2e, "
              "vs the true x %.2e\n",
              checked, worst_vs_dense, worst_sparse);
}

void test_well_conditioned_exact() {
  // Diagonally dominant, so the answer is known to near machine precision.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  double worst = 0.0;
  for (int trial = 0; trial < 30; ++trial) {
    const Int n = 20 + static_cast<Int>(rng() % 200);
    std::vector<Triplet> entries;
    std::vector<double> row_sum(sz(n), 0.0);
    for (Int i = 0; i < n; ++i) {
      for (int k = 0; k < 4; ++k) {
        const Int j = static_cast<Int>(rng() % sz(n));
        if (j >= i) continue;
        const double v = value(rng);
        entries.push_back({i, j, v});
        row_sum[sz(i)] += std::fabs(v);
        row_sum[sz(j)] += std::fabs(v);
      }
    }
    for (Int i = 0; i < n; ++i) entries.push_back({i, i, row_sum[sz(i)] + 1.0});
    const SparseMatrix lower = SparseMatrix::from_triplets(n, n, std::move(entries));
    CholeskyFactor factor;
    CHECK(factor.analyse(lower));
    CHECK(factor.factorize(lower));
    std::vector<double> x_true(sz(n));
    for (double& v : x_true) v = value(rng);
    std::vector<double> x = multiply(to_dense_symmetric(lower), x_true);
    factor.solve(&x);
    worst = std::fmax(worst, worst_difference(x, x_true));
  }
  CHECK(worst < 1e-12);
  std::printf("     30 diagonally dominant matrices: worst error %.2e\n", worst);
}

// A k x k grid Laplacian: the classic case where the order is everything. The
// natural order is a band of width k and fills it completely; minimum degree
// should need a small fraction of that.
void test_ordering_reduces_fill_on_a_grid() {
  const Int k = 40;
  const Int n = k * k;
  std::vector<Triplet> entries;
  for (Int r = 0; r < k; ++r) {
    for (Int c = 0; c < k; ++c) {
      const Int i = r * k + c;
      entries.push_back({i, i, 4.0});
      if (c > 0) entries.push_back({i, i - 1, -1.0});
      if (r > 0) entries.push_back({i, i - k, -1.0});
    }
  }
  const SparseMatrix lower = SparseMatrix::from_triplets(n, n, std::move(entries));

  const std::vector<Int> perm = anukulan::approximate_minimum_degree(lower);
  std::vector<char> seen(sz(n), 0);
  bool is_permutation = static_cast<Int>(perm.size()) == n;
  for (const Int i : perm) {
    if (i < 0 || i >= n || seen[sz(i)]) {
      is_permutation = false;
      break;
    }
    seen[sz(i)] = 1;
  }
  CHECK(is_permutation);

  CholeskyFactor amd;
  CholeskyFactor natural;
  CholeskyOptions natural_options;
  natural_options.ordering = CholeskyOptions::Ordering::kNatural;
  CHECK(amd.analyse(lower));
  CHECK(natural.analyse(lower, natural_options));
  CHECK(amd.exact_nonzeros() * 2 < natural.exact_nonzeros());
  std::printf("     %dx%d grid: nonzeros in L %lld with minimum degree, %lld natural\n",
              k, k, amd.exact_nonzeros(), natural.exact_nonzeros());

  CHECK(amd.factorize(lower));
  std::vector<double> x_true(sz(n));
  for (Int i = 0; i < n; ++i) x_true[sz(i)] = std::sin(0.37 * i);
  std::vector<double> x(sz(n), 0.0);
  const SparseMatrix full = [&]() {
    std::vector<Triplet> t;
    for (Int i = 0; i < n; ++i)
      for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
        const Int j = lower.index()[sz(e)];
        t.push_back({i, j, lower.value()[sz(e)]});
        if (j != i) t.push_back({j, i, lower.value()[sz(e)]});
      }
    return SparseMatrix::from_triplets(n, n, std::move(t));
  }();
  full.multiply(x_true.data(), x.data());
  amd.solve(&x);
  CHECK(worst_difference(x, x_true) < 1e-10);
}

// Dependent rows: A has two identical rows, so A A' is singular. The
// factorisation must not fail, must say it dropped a pivot, and must still
// solve a consistent right-hand side.
void test_dependent_rows_are_dropped_not_fatal() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  const Int rows = 30;
  const Int cols = 45;
  std::vector<std::vector<double>> a(sz(rows), std::vector<double>(sz(cols), 0.0));
  for (Int i = 0; i < rows; ++i)
    for (Int j = 0; j < cols; ++j)
      if (rng() % 4 == 0) a[sz(i)][sz(j)] = value(rng);
  a[7] = a[3];                  // row 7 repeats row 3
  for (Int j = 0; j < cols; ++j)  // row 12 is the sum of rows 1 and 2
    a[12][sz(j)] = a[1][sz(j)] + a[2][sz(j)];

  std::vector<Triplet> entries;
  for (Int i = 0; i < rows; ++i) {
    for (Int k = 0; k <= i; ++k) {
      double s = 0.0;
      for (Int j = 0; j < cols; ++j) s += a[sz(i)][sz(j)] * a[sz(k)][sz(j)];
      if (s != 0.0 || i == k) entries.push_back({i, k, s});
    }
  }
  const SparseMatrix lower = SparseMatrix::from_triplets(rows, rows, std::move(entries));
  CholeskyFactor factor;
  CHECK(factor.analyse(lower));
  CHECK(factor.factorize(lower));
  CHECK(factor.dropped_pivots() >= 2);

  // b = A A' w is in the range, so M x = b has solutions and the factor must
  // produce one of them.
  std::vector<double> w(sz(rows));
  for (double& v : w) v = value(rng);
  const Dense m = to_dense_symmetric(lower);
  const std::vector<double> b = multiply(m, w);
  std::vector<double> x = b;
  factor.solve(&x);
  const std::vector<double> mx = multiply(m, x);
  double residual = 0.0;
  double scale = 0.0;
  for (Int i = 0; i < rows; ++i) {
    residual = std::fmax(residual, std::fabs(mx[sz(i)] - b[sz(i)]));
    scale = std::fmax(scale, std::fabs(b[sz(i)]));
  }
  CHECK(residual <= 1e-9 * scale);
  bool finite = true;
  for (const double v : x) finite = finite && std::isfinite(v);
  CHECK(finite);
  std::printf("     dependent rows: %d pivots dropped, residual %.2e of %.2e\n",
              factor.dropped_pivots(), residual, scale);
}

void test_negative_pivot_is_regularised() {
  // Indefinite: [[1, 2], [2, 1]] has a negative second pivot.
  std::vector<Triplet> entries{{0, 0, 1.0}, {1, 0, 2.0}, {1, 1, 1.0}, {2, 2, 3.0}};
  const SparseMatrix lower = SparseMatrix::from_triplets(3, 3, std::move(entries));
  CholeskyFactor factor;
  CholeskyOptions options;
  options.ordering = CholeskyOptions::Ordering::kNatural;
  CHECK(factor.analyse(lower, options));
  CHECK(factor.factorize(lower));
  CHECK_EQ(factor.dropped_pivots(), 1);
  std::vector<double> x{1.0, 1.0, 3.0};
  factor.solve(&x);
  for (const double v : x) CHECK(std::isfinite(v));
  CHECK_NEAR(x[2], 1.0, 1e-14);
}

void test_new_values_same_pattern() {
  // The interior point refactorises the same pattern every iteration.
  std::mt19937 rng(3);
  const SparseMatrix first = normal_matrix(&rng, 50, 80, 0.08, 1.0);
  CholeskyFactor factor;
  CHECK(factor.analyse(first));
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  for (int round = 0; round < 4; ++round) {
    SparseMatrix m = first;
    for (Int i = 0; i < m.rows(); ++i)
      for (Int e = m.row_begin(i); e < m.row_end(i); ++e)
        if (m.index()[sz(e)] == i) m.value()[sz(e)] += 1.0 + round;
    CHECK(factor.factorize(m));
    std::vector<double> x_true(50);
    for (double& v : x_true) v = value(rng);
    std::vector<double> x = multiply(to_dense_symmetric(m), x_true);
    std::vector<double> reference = x;
    CHECK(dense_solve(to_dense_symmetric(m), &reference));
    factor.solve(&x);
    CHECK(worst_difference(x, reference) < 1e-8);
  }
}

void test_trivial_sizes() {
  CholeskyFactor factor;
  const SparseMatrix empty = SparseMatrix::from_triplets(0, 0, {});
  CHECK(factor.analyse(empty));
  CHECK(factor.factorize(empty));
  std::vector<double> none;
  factor.solve(&none);

  const SparseMatrix one = SparseMatrix::from_triplets(1, 1, {{0, 0, 4.0}});
  CHECK(factor.analyse(one));
  CHECK(factor.factorize(one));
  std::vector<double> x{8.0};
  factor.solve(&x);
  CHECK_NEAR(x[0], 2.0, 1e-15);
}

}  // namespace

int main() {
  test_trivial_sizes();
  test_well_conditioned_exact();
  test_random_spd_against_dense();
  test_ordering_reduces_fill_on_a_grid();
  test_dependent_rows_are_dropped_not_fatal();
  test_negative_pivot_is_regularised();
  test_new_values_same_pattern();
  return anukulan_test::finish("cholesky");
}
