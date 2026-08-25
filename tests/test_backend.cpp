#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "sankhya/backend.hpp"

using sankhya::Int;
using sankhya::kInf;
using sankhya::LinAlgBackend;
using sankhya::SparseMatrix;
using sankhya::sz;
using sankhya::Triplet;

namespace {

// This file is the contract a CUDA backend has to satisfy. Every operation is
// checked against a plain, obviously-correct expression written out longhand
// here, so a second implementation can be dropped in and held to exactly the
// same standard without anyone having to reason about the solver.

const LinAlgBackend& b() { return sankhya::cpu_backend(); }

SparseMatrix make_matrix() {
  std::vector<Triplet> t = {
      {0, 0, 1.5}, {0, 2, -2.0}, {1, 1, 3.25}, {2, 0, 4.0}, {2, 3, -0.5}, {3, 2, 7.0},
  };
  return SparseMatrix::from_triplets(4, 4, std::move(t));
}

void test_multiply_matches_longhand() {
  const SparseMatrix a = make_matrix();
  const SparseMatrix at = a.transpose();
  const double x[4] = {1.0, -2.0, 3.0, 4.0};

  double got[4] = {9, 9, 9, 9};
  b().multiply(a, x, got);
  for (Int i = 0; i < 4; ++i) {
    double want = 0.0;
    for (Int k = a.row_begin(i); k < a.row_end(i); ++k)
      want += a.value()[sz(k)] * x[a.index()[sz(k)]];
    CHECK_NEAR(got[i], want, 0.0);
  }

  double gotT[4] = {9, 9, 9, 9};
  b().multiply_transpose(at, x, gotT);
  for (Int j = 0; j < 4; ++j) {
    double want = 0.0;
    for (Int i = 0; i < 4; ++i) {
      for (Int k = a.row_begin(i); k < a.row_end(i); ++k)
        if (a.index()[sz(k)] == j) want += a.value()[sz(k)] * x[i];
    }
    CHECK_NEAR(gotT[j], want, 1e-15);
  }
}

void test_reductions() {
  const double a[5] = {1.0, -2.5, 3.0, 0.0, -7.25};
  const double c[5] = {2.0, 4.0, -1.0, 9.0, 0.5};
  double want_dot = 0.0, want_sq = 0.0, want_inf = 0.0;
  for (int i = 0; i < 5; ++i) {
    want_dot += a[i] * c[i];
    want_sq += a[i] * a[i];
    want_inf = std::fmax(want_inf, std::fabs(a[i]));
  }
  CHECK_NEAR(b().dot(a, c, 5), want_dot, 1e-15);
  CHECK_NEAR(b().two_norm(a, 5), std::sqrt(want_sq), 1e-15);
  CHECK_NEAR(b().inf_norm(a, 5), want_inf, 0.0);

  const double empty[1] = {0.0};
  CHECK_NEAR(b().dot(empty, empty, 0), 0.0, 0.0);
  CHECK_NEAR(b().inf_norm(empty, 0), 0.0, 0.0);
}

void test_primal_step_projects_and_derives() {
  const Int n = 5;
  const double x[5] = {0.0, 1.0, -1.0, 5.0, 2.0};
  const double c[5] = {1.0, -1.0, 0.5, 2.0, 0.0};
  const double kty[5] = {0.5, 0.5, -0.5, 1.0, 3.0};
  const double lower[5] = {0.0, -kInf, -2.0, 0.0, 1.0};
  const double upper[5] = {1.0, 1.0, kInf, 4.0, 1.5};
  const double tau = 0.75;

  double xn[5], dx[5], xb[5];
  b().primal_step(n, tau, x, c, kty, lower, upper, xn, dx, xb);

  for (Int j = 0; j < n; ++j) {
    double want = x[sz(j)] - tau * (c[sz(j)] - kty[sz(j)]);
    if (want < lower[sz(j)]) want = lower[sz(j)];
    if (want > upper[sz(j)]) want = upper[sz(j)];
    CHECK_NEAR(xn[sz(j)], want, 0.0);
    CHECK_NEAR(dx[sz(j)], want - x[sz(j)], 0.0);
    CHECK_NEAR(xb[sz(j)], 2.0 * want - x[sz(j)], 1e-15);
    CHECK(xn[sz(j)] >= lower[sz(j)] - 1e-15);
    CHECK(xn[sz(j)] <= upper[sz(j)] + 1e-15);
  }
}

void test_dual_step_splits_at_the_equality_block() {
  const Int m = 5;
  const Int equalities = 2;
  const double y[5] = {1.0, -1.0, 0.5, -0.5, 2.0};
  const double q[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
  const double kxb[5] = {3.0, 3.0, 3.0, 3.0, 3.0};
  const double kx[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
  const double sigma = 0.5;

  double yn[5], dy[5], kdx[5];
  b().dual_step(m, equalities, sigma, y, q, kxb, kx, yn, dy, kdx);

  for (Int i = 0; i < m; ++i) {
    double want = y[sz(i)] + sigma * (q[sz(i)] - kxb[sz(i)]);
    if (i >= equalities && want < 0.0) want = 0.0;
    CHECK_NEAR(yn[sz(i)], want, 0.0);
    CHECK_NEAR(dy[sz(i)], want - y[sz(i)], 0.0);
    CHECK_NEAR(kdx[sz(i)], 0.5 * (kxb[sz(i)] - kx[sz(i)]), 0.0);
  }
  // The equality block keeps its sign; the inequality block does not.
  CHECK(yn[1] < 0.0);
  CHECK(yn[3] >= 0.0);
  CHECK(yn[4] >= 0.0);
}

void test_kdx_identity_holds_against_a_real_matrix() {
  // k_dx is derived rather than computed, using K x_bar = 2 K x_next - K x. If
  // that identity were wrong the adaptive step size would be driven by a
  // quantity that is not K(x_next - x), and nothing would report it.
  const SparseMatrix a = make_matrix();
  const double x[4] = {0.5, -1.0, 2.0, 1.0};
  const double x_next[4] = {1.5, -0.5, 1.0, 2.0};
  double x_bar[4], k_x[4], k_x_bar[4], k_dx_direct[4];
  for (int j = 0; j < 4; ++j) x_bar[j] = 2.0 * x_next[j] - x[j];
  b().multiply(a, x, k_x);
  b().multiply(a, x_bar, k_x_bar);

  double diff[4];
  for (int j = 0; j < 4; ++j) diff[j] = x_next[j] - x[j];
  b().multiply(a, diff, k_dx_direct);

  double yn[4], dy[4], kdx[4];
  const double y[4] = {0, 0, 0, 0};
  const double q[4] = {0, 0, 0, 0};
  b().dual_step(4, 4, 0.0, y, q, k_x_bar, k_x, yn, dy, kdx);
  for (int i = 0; i < 4; ++i) CHECK_NEAR(kdx[i], k_dx_direct[i], 1e-13);
}

void test_accumulate_and_average() {
  const Int n = 4;
  double sum[4] = {0, 0, 0, 0};
  const double a[4] = {1.0, 2.0, 3.0, 4.0};
  const double c[4] = {0.5, 0.5, 0.5, 0.5};
  b().accumulate(n, 2.0, a, sum);
  b().accumulate(n, 3.0, c, sum);
  for (Int j = 0; j < n; ++j)
    CHECK_NEAR(sum[sz(j)], 2.0 * a[sz(j)] + 3.0 * c[sz(j)], 1e-15);

  double out[4];
  b().scale_into(n, 5.0, sum, out);
  for (Int j = 0; j < n; ++j) CHECK_NEAR(out[sz(j)], sum[sz(j)] / 5.0, 0.0);
}

void test_weighted_norm() {
  const double dx[3] = {1.0, -2.0, 0.5};
  const double dy[2] = {3.0, -1.0};
  for (const double omega : {0.25, 1.0, 4.0}) {
    double want = 0.0;
    for (int j = 0; j < 3; ++j) want += omega * dx[j] * dx[j];
    for (int i = 0; i < 2; ++i) want += dy[i] * dy[i] / omega;
    CHECK_NEAR(b().weighted_norm_squared(3, 2, dx, dy, omega), want, 1e-14);
  }
}

void test_large_random_consistency() {
  // A size where a GPU implementation would actually be exercised, checked
  // against longhand arithmetic.
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<double> pick(-3.0, 3.0);
  const Int n = 2000;
  std::vector<Triplet> t;
  for (Int i = 0; i < n; ++i) {
    for (int k = 0; k < 5; ++k) {
      t.push_back(Triplet{i, static_cast<Int>((i * 7 + k * 311) % n), pick(rng)});
    }
  }
  const SparseMatrix a = SparseMatrix::from_triplets(n, n, std::move(t));
  std::vector<double> x(sz(n));
  for (double& v : x) v = pick(rng);

  std::vector<double> got(sz(n));
  b().multiply(a, x.data(), got.data());
  for (Int i = 0; i < n; i += 137) {
    double want = 0.0;
    for (Int k = a.row_begin(i); k < a.row_end(i); ++k)
      want += a.value()[sz(k)] * x[sz(a.index()[sz(k)])];
    CHECK_NEAR(got[sz(i)], want, 1e-13);
  }
  CHECK(std::isfinite(b().two_norm(got.data(), n)));
}

}  // namespace

int main() {
  test_multiply_matches_longhand();
  test_reductions();
  test_primal_step_projects_and_derives();
  test_dual_step_splits_at_the_equality_block();
  test_kdx_identity_holds_against_a_real_matrix();
  test_accumulate_and_average();
  test_weighted_norm();
  test_large_random_consistency();
  return sankhya_test::finish("test_backend");
}
