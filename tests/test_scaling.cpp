#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "check.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/scaling.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::Int;
using sankhya::kInf;
using sankhya::Model;
using sankhya::Norm;
using sankhya::Scaling;
using sankhya::ScalingOptions;
using sankhya::ScalingReport;
using sankhya::StandardFormResult;
using sankhya::StandardLp;
using sankhya::sz;

namespace {

Model read_model(const std::string& text) {
  std::istringstream in(text);
  const sankhya::MpsReadResult r = sankhya::read_mps_stream(in, "<test>");
  if (!r.ok) std::fprintf(stderr, "read failed: %s\n", r.error.c_str());
  CHECK(r.ok);
  return r.model;
}

StandardLp build(const std::string& text) {
  const StandardFormResult r = sankhya::to_standard_form(read_model(text));
  CHECK(r.ok);
  return r.lp;
}

// Coefficients deliberately spanning ten orders of magnitude, which is the
// situation preconditioning exists for. Netlib has real instances like this.
const char* kBadlyScaled =
    "NAME          NASTY\n"
    "ROWS\n"
    " N  COST\n"
    " E  BIG\n"
    " G  SMALL\n"
    " L  MIXED\n"
    "COLUMNS\n"
    "    X         COST      1.0e-3     BIG       1.0e6\n"
    "    X         SMALL     1.0e-4     MIXED     1.0e2\n"
    "    Y         COST      1.0e4      BIG       5.0e5\n"
    "    Y         MIXED     3.0e-3\n"
    "    Z         COST      2.0        SMALL     7.0e-5\n"
    "    Z         MIXED     1.0e-2\n"
    "RHS\n"
    "    RHS       BIG       1.0e6      SMALL     1.0e-2\n"
    "    RHS       MIXED     4.0e2\n"
    "BOUNDS\n"
    " UP BND       X         1.0e3\n"
    " UP BND       Y         2.0e-2\n"
    " FR BND       Z\n"
    "ENDATA\n";

double max_norm(const StandardLp& lp, bool rows, Norm norm) {
  std::vector<double> out(sz(rows ? lp.k.rows() : lp.k.cols()), 0.0);
  if (rows) {
    lp.k.row_norms(norm, out.data());
  } else {
    lp.k.col_norms(norm, out.data());
  }
  double hi = 0.0;
  for (const double v : out) hi = std::fmax(hi, v);
  return hi;
}

double min_nonzero_norm(const StandardLp& lp, bool rows, Norm norm) {
  std::vector<double> out(sz(rows ? lp.k.rows() : lp.k.cols()), 0.0);
  if (rows) {
    lp.k.row_norms(norm, out.data());
  } else {
    lp.k.col_norms(norm, out.data());
  }
  double lo = kInf;
  for (const double v : out) {
    if (v > 0.0) lo = std::fmin(lo, v);
  }
  return lo;
}

void test_reduces_spread() {
  StandardLp lp = build(kBadlyScaled);
  const double before_rows = max_norm(lp, true, Norm::kInfinity) /
                             min_nonzero_norm(lp, true, Norm::kInfinity);
  // Row norms in the fixture run from 1e-4 to 1e6, so a spread of about 1e10.
  CHECK(before_rows > 1e9);

  const ScalingReport report = sankhya::scale_lp(&lp);
  std::printf("     scaling: row spread %.3e -> %.3e, col spread %.3e -> %.3e\n",
              report.row_spread_before, report.row_spread_after,
              report.col_spread_before, report.col_spread_after);
  CHECK_NEAR(report.row_spread_before, before_rows, 1e-9);
  CHECK(report.row_spread_after < 1e2);
  CHECK(report.col_spread_after < 1e2);
  CHECK(report.row_spread_after < report.row_spread_before / 1e6);

  // Ruiz drives every row and column norm towards one. Pock-Chambolle runs after
  // it in the 1-norm, so the infinity norms end up near one rather than exactly
  // at it - a factor of a few either side is what to expect.
  CHECK(max_norm(lp, true, Norm::kInfinity) < 10.0);
  CHECK(min_nonzero_norm(lp, true, Norm::kInfinity) > 0.01);
}

void test_scaled_problem_is_equivalent() {
  // The whole point: solving the scaled problem and mapping back has to give
  // exactly the original problem's answer. Checked on random points, since the
  // relationship must hold everywhere and not just at the optimum.
  const StandardLp original = build(kBadlyScaled);
  StandardLp scaled = original;
  const ScalingReport report = sankhya::scale_lp(&scaled);
  const Scaling& s = report.scaling;

  std::mt19937 rng(20260823);
  std::uniform_real_distribution<double> pick(-1e3, 1e3);
  std::vector<double> scratch_a;
  std::vector<double> scratch_b;

  for (int trial = 0; trial < 5000; ++trial) {
    std::vector<double> x(sz(original.num_cols()));
    for (double& v : x) v = pick(rng);

    // x_tilde = D2^-1 x, and unscale_primal must invert that exactly.
    std::vector<double> x_tilde(x.size());
    for (std::size_t j = 0; j < x.size(); ++j) x_tilde[j] = x[j] / s.col_scale[j];
    std::vector<double> back;
    s.unscale_primal(x_tilde, &back);
    for (std::size_t j = 0; j < x.size(); ++j) CHECK_NEAR(back[j], x[j], 1e-12);

    // Objective is invariant under the change of variable.
    CHECK_NEAR(scaled.standard_objective(x_tilde), original.standard_objective(x),
               1e-9);

    // Bounds agree: x is inside the original box exactly when x_tilde is inside
    // the scaled one.
    bool in_original = true;
    for (std::size_t j = 0; j < x.size(); ++j) {
      if (x[j] < original.lower[j] || x[j] > original.upper[j]) in_original = false;
    }
    bool in_scaled = true;
    for (std::size_t j = 0; j < x.size(); ++j) {
      if (x_tilde[j] < scaled.lower[j] || x_tilde[j] > scaled.upper[j])
        in_scaled = false;
    }
    CHECK_EQ(in_original, in_scaled);

    // Row activities line up row by row: K_tilde x_tilde = D1 K x, so a
    // violation in one is a violation in the other, up to the positive scale.
    std::vector<double> kx(sz(original.num_rows()));
    std::vector<double> ktx(sz(scaled.num_rows()));
    original.k.multiply(x.data(), kx.data());
    scaled.k.multiply(x_tilde.data(), ktx.data());
    for (Int i = 0; i < original.num_rows(); ++i) {
      CHECK_NEAR(ktx[sz(i)], s.row_scale[sz(i)] * kx[sz(i)], 1e-6);
    }

    double inf_original = 0.0;
    double inf_scaled = 0.0;
    original.primal_residual(x, &scratch_a, nullptr, &inf_original);
    scaled.primal_residual(x_tilde, &scratch_b, nullptr, &inf_scaled);
    CHECK_EQ(inf_original > 0.0, inf_scaled > 0.0);
  }
}

void test_infinite_bounds_survive() {
  StandardLp lp = build(kBadlyScaled);
  const StandardLp before = lp;
  sankhya::scale_lp(&lp);
  for (std::size_t j = 0; j < before.lower.size(); ++j) {
    CHECK_EQ(std::isinf(before.lower[j]), std::isinf(lp.lower[j]));
    CHECK_EQ(std::isinf(before.upper[j]), std::isinf(lp.upper[j]));
    if (std::isinf(before.lower[j])) CHECK(lp.lower[j] < 0.0);
    if (std::isinf(before.upper[j])) CHECK(lp.upper[j] > 0.0);
  }
}

void test_transpose_follows_the_matrix() {
  // PDHG reads K and K' every iteration. If scaling updates one and not the
  // other, the dual update is quietly wrong and nothing ever reports it.
  StandardLp lp = build(kBadlyScaled);
  sankhya::scale_lp(&lp);

  std::mt19937 rng(5);
  std::uniform_real_distribution<double> pick(-2.0, 2.0);
  std::vector<double> y(sz(lp.num_rows()));
  for (double& v : y) v = pick(rng);

  std::vector<double> via_cached(sz(lp.num_cols()));
  std::vector<double> direct(sz(lp.num_cols()));
  lp.kt.multiply(y.data(), via_cached.data());
  lp.k.multiply_transpose(y.data(), direct.data());
  for (std::size_t j = 0; j < direct.size(); ++j)
    CHECK_NEAR(via_cached[j], direct[j], 1e-13);

  std::string error;
  CHECK(lp.validate(&error));
}

void test_scaling_is_stable_when_repeated() {
  // A second pass over an already equilibrated matrix must not drift or blow up.
  StandardLp once = build(kBadlyScaled);
  sankhya::scale_lp(&once);
  const double spread_once = max_norm(once, true, Norm::kInfinity) /
                             min_nonzero_norm(once, true, Norm::kInfinity);

  StandardLp twice = once;
  sankhya::scale_lp(&twice);
  const double spread_twice = max_norm(twice, true, Norm::kInfinity) /
                              min_nonzero_norm(twice, true, Norm::kInfinity);
  CHECK(spread_twice < spread_once * 10.0);
  CHECK(std::isfinite(spread_twice));
}

void test_empty_rows_and_columns_are_left_alone() {
  // Presolve leaves empty rows and columns behind, and dividing by a zero norm
  // would turn the whole matrix into NaN.
  const char* text =
      "NAME          EMPTY\n"
      "ROWS\n"
      " N  COST\n"
      " G  USED\n"
      " G  UNUSED\n"
      "COLUMNS\n"
      "    X         COST      1.0        USED      2.0\n"
      "    IDLE      COST      1.0\n"
      "RHS\n"
      "    RHS       USED      1.0        UNUSED    -5.0\n"
      "ENDATA\n";
  StandardLp lp = build(text);
  const ScalingReport report = sankhya::scale_lp(&lp);
  for (const double v : report.scaling.row_scale) CHECK(std::isfinite(v) && v > 0.0);
  for (const double v : report.scaling.col_scale) CHECK(std::isfinite(v) && v > 0.0);
  for (const double v : lp.k.value()) CHECK(std::isfinite(v));
  for (const double v : lp.q) CHECK(std::isfinite(v));
  for (const double v : lp.c) CHECK(std::isfinite(v));
}

void test_ruiz_only_option() {
  StandardLp with_pc = build(kBadlyScaled);
  StandardLp ruiz_only = build(kBadlyScaled);
  ScalingOptions options;
  options.pock_chambolle = false;
  const ScalingReport a = sankhya::scale_lp(&with_pc);
  const ScalingReport b = sankhya::scale_lp(&ruiz_only, options);

  // Ruiz on its own drives the infinity norms very close to one; the extra
  // Pock-Chambolle pass trades some of that for better 1-norm behaviour.
  CHECK(b.row_spread_after < 1.5);
  CHECK(a.row_spread_after < 1e2);
}

}  // namespace

int main() {
  test_reduces_spread();
  test_scaled_problem_is_equivalent();
  test_infinite_bounds_survive();
  test_transpose_follows_the_matrix();
  test_scaling_is_stable_when_repeated();
  test_empty_rows_and_columns_are_left_alone();
  test_ruiz_only_option();
  return sankhya_test::finish("test_scaling");
}
