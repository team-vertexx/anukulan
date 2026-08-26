// LDL' either reproduces the matrix it factorised or it does not, and the only
// way to know is to multiply back. Every test here builds a quasi-definite K,
// picks an x, forms b = K x, solves, and compares - never checks a residual the
// factorisation computed about itself.
#include "sankhya/ldl.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"

using sankhya::Int;
using sankhya::LdlFactor;
using sankhya::SparseMatrix;
using sankhya::Triplet;
using sankhya::sz;

namespace {

// Full symmetric matrix from the lower triangle, so the product can be taken
// without the factorisation's own conventions getting a say.
SparseMatrix full_from_lower(const SparseMatrix& lower) {
  std::vector<Triplet> entries;
  for (Int i = 0; i < lower.rows(); ++i) {
    for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
      const Int j = lower.index()[sz(e)];
      const double v = lower.value()[sz(e)];
      if (j > i) continue;
      entries.push_back({i, j, v});
      if (j != i) entries.push_back({j, i, v});
    }
  }
  return SparseMatrix::from_triplets(lower.rows(), lower.cols(), std::move(entries));
}

double worst_solve_error(const SparseMatrix& lower, const LdlFactor& factor,
                         const std::vector<double>& x_true) {
  const SparseMatrix full = full_from_lower(lower);
  std::vector<double> b(sz(full.rows()), 0.0);
  full.multiply(x_true.data(), b.data());
  std::vector<double> x = b;
  factor.solve(&x);
  double worst = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i)
    worst = std::fmax(worst, std::fabs(x[i] - x_true[i]));
  return worst;
}

// [ P + sigma I   A' ]
// [ A          -I/rho ]
SparseMatrix build_kkt(Int n, Int m, const std::vector<Triplet>& p_upper,
                       const std::vector<Triplet>& a, double sigma, double rho) {
  std::vector<Triplet> entries;
  // Lower triangle, row-wise: row r holds columns c <= r.
  for (const Triplet& t : p_upper) {
    if (t.col > t.row) entries.push_back({t.col, t.row, t.value});
    else entries.push_back(t);
  }
  for (Int j = 0; j < n; ++j) entries.push_back({j, j, sigma});
  // A sits in the lower left, so row n+i column j.
  for (const Triplet& t : a) entries.push_back({n + t.row, t.col, t.value});
  for (Int i = 0; i < m; ++i) entries.push_back({n + i, n + i, -1.0 / rho});
  return SparseMatrix::from_triplets(n + m, n + m, std::move(entries));
}

void test_diagonal() {
  // The simplest quasi-definite matrix there is, where the answer is obvious.
  std::vector<Triplet> entries;
  for (Int i = 0; i < 6; ++i) entries.push_back({i, i, i < 3 ? 2.0 : -3.0});
  const SparseMatrix k = SparseMatrix::from_triplets(6, 6, std::move(entries));
  LdlFactor f;
  std::string error;
  CHECK(f.analyse(k, &error));
  CHECK(f.factorize(k, {}, &error));
  CHECK_EQ(f.positive_pivots(), 3);
  const std::vector<double> x{1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
  CHECK(worst_solve_error(k, f, x) < 1e-14);
}

void test_small_kkt_by_hand() {
  //  P = [[2,1],[1,3]],  A = [[1,1],[1,-1]],  sigma 1e-6, rho 0.1
  const std::vector<Triplet> p{{0, 0, 2.0}, {0, 1, 1.0}, {1, 1, 3.0}};
  const std::vector<Triplet> a{{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}};
  const SparseMatrix k = build_kkt(2, 2, p, a, 1e-6, 0.1);
  LdlFactor f;
  std::string error;
  CHECK(f.analyse(k, &error));
  CHECK(f.factorize(k, {}, &error));
  // Two primal variables, so two positive pivots and two negative.
  CHECK_EQ(f.positive_pivots(), 2);
  const std::vector<double> x{0.5, -1.25, 3.0, 2.0};
  const double e = worst_solve_error(k, f, x);
  CHECK(e < 1e-12);
  std::printf("     2x2 KKT by hand: worst solve error %.2e\n", e);
}

void test_random_kkt() {
  std::mt19937 rng(20260826);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  double worst = 0.0;
  double worst_fill = 0.0;
  int solved = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Int n = 5 + static_cast<Int>(rng() % 25);
    const Int m = 3 + static_cast<Int>(rng() % 20);

    // P = B'B + I keeps the leading block positive definite whatever B is.
    std::vector<std::vector<double>> dense(sz(n), std::vector<double>(sz(n), 0.0));
    for (Int r = 0; r < n; ++r)
      for (Int c = 0; c < n; ++c)
        if (unit(rng) < 0.3) dense[sz(r)][sz(c)] = value(rng);
    std::vector<Triplet> p;
    for (Int i = 0; i < n; ++i) {
      for (Int j = i; j < n; ++j) {
        double s = (i == j) ? 1.0 : 0.0;
        for (Int t = 0; t < n; ++t) s += dense[sz(t)][sz(i)] * dense[sz(t)][sz(j)];
        if (std::fabs(s) > 1e-12) p.push_back({i, j, s});
      }
    }
    std::vector<Triplet> a;
    for (Int i = 0; i < m; ++i)
      for (Int j = 0; j < n; ++j)
        if (unit(rng) < 0.35) a.push_back({i, j, value(rng)});

    const SparseMatrix k = build_kkt(n, m, p, a, 1e-6, 0.1);
    LdlFactor f;
    std::string error;
    if (!f.analyse(k, &error)) { CHECK(false); continue; }
    if (!f.factorize(k, {}, &error)) {
      sankhya_test::report(__FILE__, __LINE__, "factorize failed: " + error);
      continue;
    }
    // Quasi-definite: exactly n positive pivots and m negative, whatever the
    // ordering. A different count means it factorised something else.
    CHECK_EQ(f.positive_pivots(), n);

    std::vector<double> x(sz(n + m));
    for (double& v : x) v = value(rng);
    worst = std::fmax(worst, worst_solve_error(k, f, x));
    worst_fill = std::fmax(worst_fill, f.fill_ratio());
    ++solved;
  }
  CHECK(solved >= 35);
  CHECK(worst < 1e-8);
  std::printf("     %d random KKT systems: worst solve error %.2e, worst fill %.2f\n",
              solved, worst, worst_fill);
}

void test_values_can_change_without_reanalysing() {
  // The point of a quasi-definite matrix: one ordering, many sets of numbers.
  // ADMM changes rho and refactorises without touching the pattern.
  const std::vector<Triplet> p{{0, 0, 4.0}, {0, 2, 1.0}, {1, 1, 3.0}, {2, 2, 5.0}};
  const std::vector<Triplet> a{{0, 0, 1.0}, {0, 2, 2.0}, {1, 1, 1.0}};
  LdlFactor f;
  std::string error;
  const SparseMatrix first = build_kkt(3, 2, p, a, 1e-6, 0.1);
  CHECK(f.analyse(first, &error));
  CHECK(f.factorize(first, {}, &error));
  const std::vector<double> x{1.0, 2.0, 3.0, -1.0, -2.0};
  CHECK(worst_solve_error(first, f, x) < 1e-12);

  // Same pattern, different rho. No second analyse().
  const SparseMatrix second = build_kkt(3, 2, p, a, 1e-6, 25.0);
  CHECK(f.factorize(second, {}, &error));
  const double e = worst_solve_error(second, f, x);
  CHECK(e < 1e-12);
  std::printf("     refactorised with a new rho, no re-analysis: error %.2e\n", e);
}

}  // namespace

int main() {
  test_diagonal();
  test_small_kkt_by_hand();
  test_random_kkt();
  test_values_can_change_without_reanalysing();
  return sankhya_test::finish("ldl");
}
