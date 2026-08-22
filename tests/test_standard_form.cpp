#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "check.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::kInf;
using sankhya::Model;
using sankhya::StandardFormResult;
using sankhya::StandardLp;

namespace {

Model read_model(const std::string& text) {
  std::istringstream in(text);
  const sankhya::MpsReadResult r = sankhya::read_mps_stream(in, "<test>");
  if (!r.ok) std::fprintf(stderr, "read failed: %s\n", r.error.c_str());
  CHECK(r.ok);
  return r.model;
}

// One row of every kind: equality, <=, >=, two-sided and free.
const char* kAllRowKinds =
    "NAME          KINDS\n"
    "ROWS\n"
    " N  COST\n"
    " E  REQ\n"
    " L  RLE\n"
    " G  RGE\n"
    " L  RRANGE\n"
    "COLUMNS\n"
    "    X         COST      1.0        REQ       1.0\n"
    "    X         RLE       2.0        RGE       3.0\n"
    "    X         RRANGE    4.0\n"
    "    Y         COST      2.0        REQ       5.0\n"
    "    Y         RLE       6.0        RRANGE    7.0\n"
    "RHS\n"
    "    RHS       REQ       10.0       RLE       20.0\n"
    "    RHS       RGE       30.0       RRANGE    40.0\n"
    "RANGES\n"
    "    RNG       RRANGE    15.0\n"
    "BOUNDS\n"
    " UP BND       X         100.0\n"
    " FR BND       Y\n"
    "ENDATA\n";

void test_row_splitting() {
  const Model m = read_model(kAllRowKinds);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  const StandardLp& lp = r.lp;

  CHECK_EQ(r.rows_from_equalities, 1);
  CHECK_EQ(r.rows_from_inequalities, 2);  // RLE and RGE
  CHECK_EQ(r.rows_from_ranges, 1);        // RRANGE, contributing two rows
  CHECK_EQ(lp.num_equalities, 1);
  CHECK_EQ(lp.num_rows(), 5);             // 1 + 2 + 2
  CHECK_EQ(lp.num_inequalities(), 4);

  // The equality block comes first and keeps its original sign.
  CHECK_NEAR(lp.q[0], 10.0, 0.0);
  CHECK_NEAR(lp.row_origin[0].sign, 1.0, 0.0);

  // Every inequality row is a >= row after the transform.
  for (sankhya::Int i = lp.num_equalities; i < lp.num_rows(); ++i) {
    const double sign = lp.row_origin[static_cast<std::size_t>(i)].sign;
    CHECK(sign == 1.0 || sign == -1.0);
  }

  // The <= row was negated: -6y - 2x >= -20.
  bool found_negated_le = false;
  for (sankhya::Int i = lp.num_equalities; i < lp.num_rows(); ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    if (lp.row_origin[si].model_row == 1 && lp.row_origin[si].sign == -1.0) {
      found_negated_le = true;
      CHECK_NEAR(lp.q[si], -20.0, 0.0);
    }
  }
  CHECK(found_negated_le);
}

void test_free_row_is_dropped() {
  const std::string text =
      "NAME          FREEROW\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        1.0\n"
      "RANGES\n"
      "ENDATA\n";
  Model m = read_model(text);
  // Make R1 free by hand, which is what presolve will do to redundant rows.
  m.row_lower[0] = -kInf;
  m.row_upper[0] = kInf;
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  CHECK_EQ(r.free_rows_dropped, 1);
  CHECK_EQ(r.lp.num_rows(), 0);
  CHECK(!r.warnings.empty());
}

void test_maximize_is_negated_and_recovered() {
  const std::string text =
      "NAME          MAXP\n"
      "OBJSENSE\n"
      "    MAX\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      3.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        5.0        COST      -2.0\n"
      "ENDATA\n";
  const Model m = read_model(text);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) return;

  CHECK_NEAR(r.lp.objective_scale, -1.0, 0.0);
  CHECK_NEAR(r.lp.c[0], -3.0, 0.0);       // negated for minimization
  CHECK_NEAR(r.lp.objective_offset, 2.0, 0.0);  // RHS on the objective row negates

  // At x = 5 the model objective is 3*5 + 2 = 17, and the standard form must
  // report the same thing after undoing the negation.
  const std::vector<double> x = {5.0};
  CHECK_NEAR(r.lp.standard_objective(x), -15.0, 1e-12);
  CHECK_NEAR(r.lp.model_objective(x), 17.0, 1e-12);
}

// Feasibility for the model, checked directly against the row bounds.
bool model_feasible(const Model& m, const std::vector<double>& x, double tol) {
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (x[j] < m.col_lower[j] - tol || x[j] > m.col_upper[j] + tol) return false;
  }
  std::vector<double> ax(static_cast<std::size_t>(m.num_rows()), 0.0);
  m.constraints.multiply(x.data(), ax.data());
  for (std::size_t i = 0; i < ax.size(); ++i) {
    if (ax[i] < m.row_lower[i] - tol || ax[i] > m.row_upper[i] + tol) return false;
  }
  return true;
}

// Fixture with no equality row and a roomy feasible region, so that uniform
// sampling actually lands inside it often enough to prove something.
const char* kInequalitiesOnly =
    "NAME          INEQ\n"
    "ROWS\n"
    " N  COST\n"
    " L  RLE\n"
    " G  RGE\n"
    " L  RRANGE\n"
    "COLUMNS\n"
    "    X         COST      1.0        RLE       1.0\n"
    "    X         RGE       1.0        RRANGE    2.0\n"
    "    Y         COST      2.0        RLE       1.0\n"
    "    Y         RGE       -1.0       RRANGE    1.0\n"
    "RHS\n"
    "    RHS       RLE       30.0       RGE       -20.0\n"
    "    RHS       RRANGE    50.0\n"
    "RANGES\n"
    "    RNG       RRANGE    40.0\n"
    "BOUNDS\n"
    " UP BND       X         100.0\n"
    " FR BND       Y\n"
    "ENDATA\n";

// Feasibility according to the standard form: bounds plus a zero primal residual.
bool standard_feasible(const StandardLp& lp, const std::vector<double>& x,
                       std::vector<double>* scratch, double tol) {
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (x[j] < lp.lower[j] - tol || x[j] > lp.upper[j] + tol) return false;
  }
  double inf_norm = 0.0;
  lp.primal_residual(x, scratch, nullptr, &inf_norm);
  return inf_norm <= tol;
}

// Samples random points and checks that both descriptions of the feasible set
// agree on every one of them. Returns how many of those points were feasible, so
// the caller can insist the sampling box was not simply missing the region.
int agreement_trials(const char* fixture, int trials, unsigned seed, double lo,
                     double hi) {
  const Model m = read_model(fixture);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return 0;
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> pick(lo, hi);
  std::vector<double> scratch;
  int disagreements = 0;
  int feasible_seen = 0;

  for (int trial = 0; trial < trials; ++trial) {
    std::vector<double> x(static_cast<std::size_t>(m.num_cols()));
    for (double& v : x) v = pick(rng);
    const bool want = model_feasible(m, x, 1e-9);
    const bool got = standard_feasible(r.lp, x, &scratch, 1e-9);
    if (want != got) ++disagreements;
    if (want) ++feasible_seen;
  }
  CHECK_EQ(disagreements, 0);
  return feasible_seen;
}

void test_feasible_sets_agree() {
  // The transform is only correct if it describes exactly the same feasible set.
  // Random points check that far better than any handpicked example.
  //
  // Two fixtures on purpose. The inequality-only one has a roomy feasible region
  // so that plenty of sampled points land inside it; the one with an equality row
  // has a feasible set of measure zero, so sampling there proves that infeasible
  // points are rejected consistently but can never confirm the other direction.
  const int inside = agreement_trials(kInequalitiesOnly, 20000, 20260823, -30.0, 60.0);
  CHECK(inside > 200);

  const int with_equality = agreement_trials(kAllRowKinds, 20000, 99, -30.0, 60.0);
  CHECK_EQ(with_equality, 0);  // as expected: an equality is never hit by sampling
}

void test_equality_feasible_point_is_accepted() {
  // Since sampling cannot reach the equality row's feasible set, pin it down by
  // hand. Solving kAllRowKinds gives exactly one feasible point: x = 10, y = 0.
  const Model m = read_model(kAllRowKinds);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) return;

  std::vector<double> scratch;
  const std::vector<double> feasible = {10.0, 0.0};
  CHECK(model_feasible(m, feasible, 1e-9));
  CHECK(standard_feasible(r.lp, feasible, &scratch, 1e-9));

  // Nudging off the equality must be rejected by both descriptions.
  for (const double delta : {1e-6, -1e-6, 0.5, -0.5}) {
    const std::vector<double> off = {10.0 + delta, 0.0};
    CHECK(!model_feasible(m, off, 1e-9));
    CHECK(!standard_feasible(r.lp, off, &scratch, 1e-9));
  }
}

void test_objective_agrees_with_model() {
  const Model m = read_model(kAllRowKinds);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) return;

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> pick(-10.0, 10.0);
  for (int trial = 0; trial < 500; ++trial) {
    std::vector<double> x(static_cast<std::size_t>(m.num_cols()));
    for (double& v : x) v = pick(rng);
    double direct = m.objective_offset;
    for (std::size_t j = 0; j < x.size(); ++j) direct += m.objective[j] * x[j];
    CHECK_NEAR(r.lp.model_objective(x), direct, 1e-12);
  }
}

void test_primal_residual_matches_definition() {
  const Model m = read_model(kAllRowKinds);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) return;
  const StandardLp& lp = r.lp;

  const std::vector<double> x = {2.0, -1.0};
  std::vector<double> kx(static_cast<std::size_t>(lp.num_rows()));
  lp.k.multiply(x.data(), kx.data());

  double expect_sq = 0.0;
  double expect_inf = 0.0;
  for (sankhya::Int i = 0; i < lp.num_rows(); ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    const double v = (i < lp.num_equalities) ? (kx[si] - lp.q[si])
                                             : std::fmin(kx[si] - lp.q[si], 0.0);
    expect_sq += v * v;
    expect_inf = std::fmax(expect_inf, std::fabs(v));
  }

  std::vector<double> scratch;
  double two = 0.0;
  double inf = 0.0;
  lp.primal_residual(x, &scratch, &two, &inf);
  CHECK_NEAR(two, std::sqrt(expect_sq), 1e-12);
  CHECK_NEAR(inf, expect_inf, 1e-12);
}

void test_transpose_is_consistent() {
  const Model m = read_model(kAllRowKinds);
  const StandardFormResult r = sankhya::to_standard_form(m);
  CHECK(r.ok);
  if (!r.ok) return;

  // PDHG uses K and K' every iteration; if the cached transpose disagrees with
  // the matrix, the dual update is silently wrong.
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> pick(-5.0, 5.0);
  std::vector<double> y(static_cast<std::size_t>(r.lp.num_rows()));
  for (double& v : y) v = pick(rng);

  std::vector<double> via_cached(static_cast<std::size_t>(r.lp.num_cols()));
  std::vector<double> direct(static_cast<std::size_t>(r.lp.num_cols()));
  r.lp.kt.multiply(y.data(), via_cached.data());
  r.lp.k.multiply_transpose(y.data(), direct.data());
  for (std::size_t j = 0; j < direct.size(); ++j)
    CHECK_NEAR(via_cached[j], direct[j], 1e-13);
}

}  // namespace

int main() {
  test_row_splitting();
  test_free_row_is_dropped();
  test_maximize_is_negated_and_recovered();
  test_feasible_sets_agree();
  test_equality_feasible_point_is_accepted();
  test_objective_agrees_with_model();
  test_primal_residual_matches_definition();
  test_transpose_is_consistent();
  return sankhya_test::finish("test_standard_form");
}
