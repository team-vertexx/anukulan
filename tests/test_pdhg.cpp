#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::Int;
using sankhya::kInf;
using sankhya::PdhgOptions;
using sankhya::PdhgResult;
using sankhya::PdhgStatus;
using sankhya::StandardFormResult;
using sankhya::StandardLp;
using sankhya::sz;

namespace {

StandardLp build(const std::string& text) {
  std::istringstream in(text);
  const sankhya::MpsReadResult r = sankhya::read_mps_stream(in, "<test>");
  if (!r.ok) std::fprintf(stderr, "read failed: %s\n", r.error.c_str());
  CHECK(r.ok);
  const StandardFormResult sf = sankhya::to_standard_form(r.model);
  CHECK(sf.ok);
  return sf.lp;
}

PdhgResult solve(const std::string& text, PdhgOptions options = {}) {
  options.tolerance = 1e-8;
  options.max_iterations = 200000;
  return sankhya::solve_pdhg(build(text), options);
}

// max 3x + 5y  s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x,y >= 0
// The textbook Wyndor problem. Optimum is x = 2, y = 6, objective 36.
const char* kWyndor =
    "NAME          WYNDOR\n"
    "OBJSENSE\n"
    "    MAX\n"
    "ROWS\n"
    " N  PROFIT\n"
    " L  PLANT1\n"
    " L  PLANT2\n"
    " L  PLANT3\n"
    "COLUMNS\n"
    "    X         PROFIT    3.0        PLANT1    1.0\n"
    "    X         PLANT3    3.0\n"
    "    Y         PROFIT    5.0        PLANT2    2.0\n"
    "    Y         PLANT3    2.0\n"
    "RHS\n"
    "    RHS       PLANT1    4.0        PLANT2    12.0\n"
    "    RHS       PLANT3    18.0\n"
    "ENDATA\n";

void test_known_optimum() {
  const PdhgResult r = solve(kWyndor);
  CHECK(r.status == PdhgStatus::kOptimal);
  CHECK_NEAR(r.objective, 36.0, 1e-6);
  CHECK_NEAR(r.x[0], 2.0, 1e-5);
  CHECK_NEAR(r.x[1], 6.0, 1e-5);
}

void test_equality_constraints() {
  // min x + y  s.t. x + y = 10, x - y = 2, x,y >= 0  ->  x = 6, y = 4, obj 10.
  const char* text =
      "NAME          EQS\n"
      "ROWS\n"
      " N  COST\n"
      " E  SUM\n"
      " E  DIFF\n"
      "COLUMNS\n"
      "    X         COST      1.0        SUM       1.0\n"
      "    X         DIFF      1.0\n"
      "    Y         COST      1.0        SUM       1.0\n"
      "    Y         DIFF      -1.0\n"
      "RHS\n"
      "    RHS       SUM       10.0       DIFF      2.0\n"
      "ENDATA\n";
  const PdhgResult r = solve(text);
  CHECK(r.status == PdhgStatus::kOptimal);
  CHECK_NEAR(r.objective, 10.0, 1e-6);
  CHECK_NEAR(r.x[0], 6.0, 1e-5);
  CHECK_NEAR(r.x[1], 4.0, 1e-5);
}

void test_variable_bounds_and_free_variables() {
  // min -x - y  s.t. x + y <= 10, 1 <= x <= 3, y free but capped by the row.
  // Optimum drives x to its upper bound 3 and y to 7.
  const char* text =
      "NAME          BOXED\n"
      "ROWS\n"
      " N  COST\n"
      " L  CAP\n"
      "COLUMNS\n"
      "    X         COST      -1.0       CAP       1.0\n"
      "    Y         COST      -1.0       CAP       1.0\n"
      "RHS\n"
      "    RHS       CAP       10.0\n"
      "BOUNDS\n"
      " LO BND       X         1.0\n"
      " UP BND       X         3.0\n"
      " FR BND       Y\n"
      "ENDATA\n";
  const PdhgResult r = solve(text);
  CHECK(r.status == PdhgStatus::kOptimal);
  CHECK_NEAR(r.objective, -10.0, 1e-6);
  CHECK(r.x[0] >= 1.0 - 1e-6 && r.x[0] <= 3.0 + 1e-6);
  CHECK_NEAR(r.x[0] + r.x[1], 10.0, 1e-5);
}

void test_range_row() {
  // 5 <= x + y <= 8, minimise x + 2y. Optimum sits on the lower edge: x = 5, y = 0.
  const char* text =
      "NAME          RANGED\n"
      "ROWS\n"
      " N  COST\n"
      " L  BAND\n"
      "COLUMNS\n"
      "    X         COST      1.0        BAND      1.0\n"
      "    Y         COST      2.0        BAND      1.0\n"
      "RHS\n"
      "    RHS       BAND      8.0\n"
      "RANGES\n"
      "    RNG       BAND      3.0\n"
      "ENDATA\n";
  const PdhgResult r = solve(text);
  CHECK(r.status == PdhgStatus::kOptimal);
  CHECK_NEAR(r.objective, 5.0, 1e-5);
}

void test_degenerate_problem() {
  // A genuinely degenerate optimum: three constraints active at a two
  // dimensional vertex. max x + y with x + y <= 2, x <= 1, y <= 1 forces
  // x = y = 1, and all three rows are tight there.
  //
  // The first version of this test used x + y <= 2, x - y <= 0, 2x + y <= 3,
  // whose optimal face is a whole segment rather than a point. The solver
  // returned a perfectly optimal point on that segment and the test called it a
  // failure. Multiple optima are not degeneracy.
  const char* text =
      "NAME          DEGEN\n"
      "OBJSENSE\n"
      "    MAX\n"
      "ROWS\n"
      " N  COST\n"
      " L  A\n"
      " L  B\n"
      " L  C\n"
      "COLUMNS\n"
      "    X         COST      1.0        A         1.0\n"
      "    X         B         1.0\n"
      "    Y         COST      1.0        A         1.0\n"
      "    Y         C         1.0\n"
      "RHS\n"
      "    RHS       A         2.0        B         1.0\n"
      "    RHS       C         1.0\n"
      "ENDATA\n";
  const PdhgResult r = solve(text);
  CHECK(r.status == PdhgStatus::kOptimal);
  CHECK_NEAR(r.objective, 2.0, 1e-6);
  CHECK_NEAR(r.x[0], 1.0, 1e-5);
  CHECK_NEAR(r.x[1], 1.0, 1e-5);
}

void test_objective_offset_is_carried_through() {
  const char* text =
      "NAME          OFFS\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        4.0        COST      -7.0\n"
      "ENDATA\n";
  const PdhgResult r = solve(text);
  CHECK(r.status == PdhgStatus::kOptimal);
  // min x subject to x >= 4, plus a constant of 7.
  CHECK_NEAR(r.objective, 11.0, 1e-6);
}

void test_residual_is_zero_at_a_known_optimum() {
  // The convergence measures are what the solver stops on, so they have to be
  // right independently of the solver. Wyndor's optimum and its dual are known:
  // x = (2, 6) and the binding rows are PLANT2 and PLANT3.
  const StandardLp lp = build(kWyndor);
  const PdhgResult r = solve(kWyndor);
  CHECK(r.status == PdhgStatus::kOptimal);

  const sankhya::PdhgResidual at_solution = sankhya::evaluate_residual(lp, r.x, r.y);
  CHECK(at_solution.relative_primal < 1e-7);
  CHECK(at_solution.relative_dual < 1e-7);
  CHECK(at_solution.relative_gap < 1e-7);

  // A point that is plainly not optimal must not be reported as converged.
  std::vector<double> bad_x(sz(lp.num_cols()), 0.0);
  std::vector<double> bad_y(sz(lp.num_rows()), 0.0);
  const sankhya::PdhgResidual off = sankhya::evaluate_residual(lp, bad_x, bad_y);
  CHECK(!off.converged(1e-6));
}

void test_matrix_norm_estimate() {
  // A diagonal-ish matrix whose largest singular value is known by inspection:
  // rows 3x and 4y give singular values 3 and 4, so ||K|| = 4.
  const char* text =
      "NAME          NORM\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      " G  R2\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        3.0\n"
      "    Y         COST      1.0        R2        4.0\n"
      "RHS\n"
      "    RHS       R1        1.0        R2        1.0\n"
      "ENDATA\n";
  const StandardLp lp = build(text);
  const double norm = sankhya::estimate_matrix_norm(lp, 500, 1e-12);
  CHECK_NEAR(norm, 4.0, 1e-6);
}

void test_every_configuration_still_solves() {
  // Each piece of machinery is optional, and turning any of them off must slow
  // the solver down rather than break it. This is also what the ablation in the
  // report relies on.
  struct Config {
    const char* name;
    bool adaptive;
    bool restarts;
    bool primal_weight;
    bool scaling;
  };
  const Config configs[] = {
      {"everything", true, true, true, true},
      {"no adaptive step", false, true, true, true},
      {"no restarts", true, false, true, true},
      {"no primal weight", true, true, false, true},
      {"no scaling", true, true, true, false},
      {"nothing", false, false, false, false},
  };

  for (const Config& config : configs) {
    PdhgOptions options;
    options.tolerance = 1e-6;
    options.max_iterations = 2000000;
    options.adaptive_step_size = config.adaptive;
    options.restarts = config.restarts;
    options.primal_weight_updates = config.primal_weight;
    if (!config.scaling) {
      options.scaling.ruiz_iterations = 0;
      options.scaling.pock_chambolle = false;
    }
    const PdhgResult r = sankhya::solve_pdhg(build(kWyndor), options);
    if (r.status != PdhgStatus::kOptimal) {
      std::fprintf(stderr, "config \"%s\" did not solve: %s\n", config.name,
                   sankhya::to_string(r.status).c_str());
    }
    CHECK(r.status == PdhgStatus::kOptimal);
    CHECK_NEAR(r.objective, 36.0, 1e-4);
  }
}

void test_is_deterministic() {
  // Same input, same answer, every time. Without this an ablation table means
  // nothing, and neither does a benchmark.
  const PdhgResult a = solve(kWyndor);
  const PdhgResult b = solve(kWyndor);
  CHECK_EQ(a.iterations, b.iterations);
  CHECK_EQ(a.restarts, b.restarts);
  for (std::size_t j = 0; j < a.x.size(); ++j) CHECK_NEAR(a.x[j], b.x[j], 0.0);
}

void test_iteration_limit_is_respected() {
  // The limit has to be low enough that the solver genuinely cannot finish.
  // Wyndor reaches a duality gap of 2.4e-08 by iteration 80 and machine
  // precision by 200, so an unreachable tolerance alone is not enough - the
  // first version of this test asked for 1e-14 within 200 iterations and the
  // solver simply got there.
  PdhgOptions options;
  options.tolerance = 1e-14;
  options.max_iterations = 80;
  options.termination_check_frequency = 40;
  const PdhgResult r = sankhya::solve_pdhg(build(kWyndor), options);
  CHECK(r.status == PdhgStatus::kIterationLimit);
  CHECK_EQ(r.iterations, 80);
}

void test_empty_constraint_matrix() {
  // A model whose rows carry no coefficients still has an answer: push every
  // variable to whichever bound its objective coefficient prefers.
  const char* text =
      "NAME          NOROWS\n"
      "ROWS\n"
      " N  COST\n"
      " G  EMPTY\n"
      "COLUMNS\n"
      "    X         COST      1.0\n"
      "    Y         COST      -1.0\n"
      "RHS\n"
      "    RHS       EMPTY     -1.0\n"
      "BOUNDS\n"
      " UP BND       X         5.0\n"
      " UP BND       Y         5.0\n"
      "ENDATA\n";
  const PdhgResult r = solve(text);
  CHECK_NEAR(r.x[0], 0.0, 1e-6);  // positive cost, sits at its lower bound
  CHECK_NEAR(r.x[1], 5.0, 1e-6);  // negative cost, pushed to its upper bound
  CHECK_NEAR(r.objective, -5.0, 1e-6);
}

}  // namespace

int main() {
  test_known_optimum();
  test_equality_constraints();
  test_variable_bounds_and_free_variables();
  test_range_row();
  test_degenerate_problem();
  test_objective_offset_is_carried_through();
  test_residual_is_zero_at_a_known_optimum();
  test_matrix_norm_estimate();
  test_every_configuration_still_solves();
  test_is_deterministic();
  test_iteration_limit_is_respected();
  test_empty_constraint_matrix();
  return sankhya_test::finish("test_pdhg");
}
