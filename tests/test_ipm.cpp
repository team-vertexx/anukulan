// Small LPs with known optima, built directly as a Model (row_lower/row_upper
// and col_lower/col_upper, so a free variable, a fixed one and a ranged row
// are just the bounds that say so) rather than through MPS text, which cannot
// spell most of them without a lot of ceremony.
#include <cmath>
#include <random>
#include <thread>
#include <vector>

#include "check.hpp"
#include "anukulan/crossover.hpp"
#include "anukulan/ipm.hpp"
#include "anukulan/model.hpp"
#include "anukulan/pdhg.hpp"
#include "anukulan/simplex.hpp"
#include "anukulan/sparse.hpp"
#include "anukulan/standard_form.hpp"

using anukulan::CrossoverOptions;
using anukulan::CrossoverResult;
using anukulan::Int;
using anukulan::IpmOptions;
using anukulan::IpmResult;
using anukulan::IpmStatus;
using anukulan::kInf;
using anukulan::Model;
using anukulan::ObjSense;
using anukulan::RaceOptions;
using anukulan::RaceResult;
using anukulan::SimplexOptions;
using anukulan::SimplexResult;
using anukulan::SimplexStatus;
using anukulan::StandardFormResult;
using anukulan::StandardLp;
using anukulan::sz;
using anukulan::Triplet;
using anukulan::VarType;
using anukulan::crossover_basis;
using anukulan::solve_ipm;
using anukulan::solve_lp;
using anukulan::solve_lp_race;
using anukulan::solve_pdhg;
using anukulan::to_standard_form;

namespace {

// A model with `cols` variables and no rows yet; the caller fills in
// objective, bounds and rows. Every column defaults to 0 <= x, no upper -
// callers narrow that per test.
Model empty_model(Int cols) {
  Model m;
  m.name = "test";
  m.sense = ObjSense::kMinimize;
  m.objective.assign(sz(cols), 0.0);
  m.col_lower.assign(sz(cols), 0.0);
  m.col_upper.assign(sz(cols), kInf);
  m.col_type.assign(sz(cols), VarType::kContinuous);
  m.col_names.resize(sz(cols));
  return m;
}

void add_row(Model* m, std::vector<Triplet>* entries, Int row, double lower, double upper,
            std::initializer_list<std::pair<Int, double>> terms) {
  for (const auto& [col, value] : terms) entries->push_back({row, col, value});
  if (static_cast<Int>(m->row_lower.size()) <= row) {
    m->row_lower.resize(sz(row) + 1);
    m->row_upper.resize(sz(row) + 1);
    m->row_names.resize(sz(row) + 1);
  }
  m->row_lower[sz(row)] = lower;
  m->row_upper[sz(row)] = upper;
}

StandardLp standard(const Model& m) {
  const StandardFormResult sf = to_standard_form(m);
  CHECK(sf.ok);
  return sf.lp;
}

// Runs solve_ipm and checks the objective and every bound, printing enough on
// failure to see why. `expected_status` defaults to optimal.
IpmResult check_ipm(const StandardLp& lp, double expected_objective, double tol,
                    IpmStatus expected_status = IpmStatus::kOptimal) {
  IpmOptions options;
  const IpmResult r = solve_ipm(lp, options);
  CHECK(r.status == expected_status);
  if (r.status == IpmStatus::kOptimal) {
    CHECK_NEAR(r.objective, expected_objective, tol);
    for (Int j = 0; j < lp.num_cols(); ++j) {
      CHECK(r.x[sz(j)] >= lp.lower[sz(j)] - 1e-7 * (1.0 + std::fabs(lp.lower[sz(j)])));
      CHECK(r.x[sz(j)] <= lp.upper[sz(j)] + 1e-7 * (1.0 + std::fabs(lp.upper[sz(j)])));
    }
  } else if (r.status != expected_status) {
    std::printf("     ipm status %s: %s\n", anukulan::to_string(r.status).c_str(),
                r.message.c_str());
  }
  return r;
}

void test_boxed_variables() {
  // minimize -x0 - 2 x1  s.t. x0 + x1 <= 4, 0 <= x0 <= 3, 0 <= x1 <= 3.
  // The higher-value variable is pushed to its own bound first: x1 = 3,
  // leaving x0 <= 1 from the row, so x0 = 1. Objective -1 - 6 = -7.
  Model m = empty_model(2);
  m.objective = {-1.0, -2.0};
  m.col_upper = {3.0, 3.0};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, -kInf, 4.0, {{0, 1.0}, {1, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);
  const IpmResult r = check_ipm(lp, -7.0, 1e-6);
  CHECK_NEAR(r.x[sz(0)], 1.0, 1e-5);
  CHECK_NEAR(r.x[sz(1)], 3.0, 1e-5);
}

void test_free_variable() {
  // minimize y  s.t. x + y = 5, 2 <= x <= 8, y free.
  // y = 5 - x is smallest when x is largest: x = 8, y = -3.
  Model m = empty_model(2);
  m.objective = {0.0, 1.0};
  m.col_lower = {2.0, -kInf};
  m.col_upper = {8.0, kInf};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 5.0, 5.0, {{0, 1.0}, {1, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);
  const IpmResult r = check_ipm(lp, -3.0, 1e-6);
  CHECK_NEAR(r.x[sz(0)], 8.0, 1e-5);
  CHECK_NEAR(r.x[sz(1)], -3.0, 1e-5);
}

void test_fixed_column() {
  // minimize x0 + 2 x1 + x2  s.t. x0 + x1 + x2 = 10, x2 fixed at 3, x0, x1 >= 0.
  // x0 + x1 = 7 with the cheaper variable taking all of it: x0 = 7, x1 = 0.
  // Objective 7 + 0 + 3 = 10.
  Model m = empty_model(3);
  m.objective = {1.0, 2.0, 1.0};
  m.col_lower[2] = 3.0;
  m.col_upper[2] = 3.0;
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 10.0, 10.0, {{0, 1.0}, {1, 1.0}, {2, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 3, std::move(entries));
  const StandardLp lp = standard(m);
  const IpmResult r = check_ipm(lp, 10.0, 1e-6);
  CHECK_NEAR(r.x[sz(0)], 7.0, 1e-5);
  CHECK_NEAR(r.x[sz(1)], 0.0, 1e-5);
  CHECK_NEAR(r.x[sz(2)], 3.0, 1e-9);
}

void test_ranged_row() {
  // minimize -x1  s.t. 1 <= x0 + x1 <= 5, 0 <= x0 <= 10, 0 <= x1 <= 3.
  // x1 = 3 (its own bound), x0 = 0 keeps the row inside [1, 5]. Objective -3.
  Model m = empty_model(2);
  m.objective = {0.0, -1.0};
  m.col_upper = {10.0, 3.0};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 1.0, 5.0, {{0, 1.0}, {1, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);
  const IpmResult r = check_ipm(lp, -3.0, 1e-6);
  CHECK_NEAR(r.x[sz(1)], 3.0, 1e-5);
}

void test_degenerate_optimal_face() {
  // minimize x0 + x1  s.t. x0 + x1 >= 2, 0 <= x0, x1 <= 2. Every point on the
  // segment from (0, 2) to (2, 0) is optimal at objective 2 - the interior
  // point method lands somewhere in the middle of it, crossover and the
  // simplex would land on one of its ends, and both are right. What is
  // checked is the objective and that the row is not violated, not which of
  // the many optima it is.
  Model m = empty_model(2);
  m.objective = {1.0, 1.0};
  m.col_upper = {2.0, 2.0};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 2.0, kInf, {{0, 1.0}, {1, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);
  const IpmResult r = check_ipm(lp, 2.0, 1e-6);
  CHECK(r.x[sz(0)] + r.x[sz(1)] >= 2.0 - 1e-6);
}

void test_primal_infeasibility_detected() {
  // x0 + x1 >= 10 with both capped at 3: the row can never be met.
  Model m = empty_model(2);
  m.objective = {1.0, 1.0};
  m.col_upper = {3.0, 3.0};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 10.0, kInf, {{0, 1.0}, {1, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);
  IpmOptions options;
  const IpmResult r = solve_ipm(lp, options);
  CHECK(r.status == IpmStatus::kPrimalInfeasible);
  std::printf("     primal infeasible: %s\n", r.message.c_str());
}

void test_dual_infeasibility_detected() {
  // minimize -x0 - x1  s.t. x0 - x1 = 0, x0, x1 >= 0 with no upper bound:
  // x0 = x1 = t is feasible for every t and the objective falls without
  // bound as t grows.
  Model m = empty_model(2);
  m.objective = {-1.0, -1.0};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 0.0, 0.0, {{0, 1.0}, {1, -1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);
  IpmOptions options;
  const IpmResult r = solve_ipm(lp, options);
  CHECK(r.status == IpmStatus::kDualInfeasible);
  std::printf("     dual infeasible: %s\n", r.message.c_str());
}

// A random, feasible, bounded LP - the same construction test_crossover.cpp
// uses, so the two compositions (first-order seed, interior point seed) are
// measured the same way. Feasibility is guaranteed by building the
// right-hand side from a point known to satisfy it.
Model random_lp(unsigned seed, Int rows, Int cols) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> coeff(-2.0, 3.0);
  std::uniform_real_distribution<double> point(0.0, 4.0);

  Model model = empty_model(cols);
  model.col_upper.assign(sz(cols), 10.0);
  for (double& c : model.objective) c = coeff(rng);

  std::vector<double> witness(sz(cols));
  for (Int j = 0; j < cols; ++j) witness[sz(j)] = point(rng);

  std::vector<Triplet> entries;
  model.row_lower.assign(sz(rows), -kInf);
  model.row_upper.assign(sz(rows), 0.0);
  model.row_names.resize(sz(rows));
  for (Int i = 0; i < rows; ++i) {
    double activity = 0.0;
    for (Int j = 0; j < cols; ++j) {
      if (rng() % 3 == 0) continue;
      const double v = coeff(rng);
      entries.push_back(Triplet{i, j, v});
      activity += v * witness[sz(j)];
    }
    model.row_upper[sz(i)] = activity + ((i % 2 == 0) ? 0.0 : 5.0);
  }
  model.constraints = anukulan::SparseMatrix::from_triplets(rows, cols, std::move(entries));
  return model;
}

// The interior point method does not land on a vertex; crossover and the
// simplex have to finish the job, exactly as they do for the first-order
// method today. What matters is that the finished answer agrees with a cold
// solve, on instances too irregular to have hand-picked optima.
void test_crossover_from_ipm_matches_cold_simplex() {
  Int checked = 0;
  for (unsigned seed = 1; seed <= 20; ++seed) {
    const Model model = random_lp(seed, 10 + static_cast<Int>(seed % 7),
                                  14 + static_cast<Int>(seed % 9));
    const StandardLp lp = standard(model);

    const IpmResult ipm = solve_ipm(lp);
    CHECK(ipm.status == IpmStatus::kOptimal);

    const CrossoverResult cross = crossover_basis(lp, ipm.x, ipm.y, CrossoverOptions{});
    CHECK(cross.ok);
    SimplexOptions so;
    so.start_basic = &cross.basic;
    so.start_status = &cross.status;
    const SimplexResult warm = solve_lp(lp, so);
    CHECK(warm.status == SimplexStatus::kOptimal);

    const SimplexResult cold = solve_lp(lp);
    CHECK(cold.status == SimplexStatus::kOptimal);
    CHECK_NEAR(warm.objective, cold.objective, 1e-6);
    ++checked;
  }
  std::printf("     %d random LPs: interior point crossover matches a cold simplex\n", checked);
}

// The race between the three engines, on problems small enough that every one
// of them reaches the optimum well inside a five-second cap - so the test
// bounds how long a hang could cost rather than how long the race is allowed
// to run in general. Which engine wins is a timing accident and is not
// asserted; that the winner is right is the only thing that matters.
void test_race_returns_correct_answer() {
  auto run = [](const Model& model, double expected_objective) {
    const StandardLp lp = standard(model);
    RaceOptions options;
    options.time_limit_seconds = 5.0;
    const RaceResult r = solve_lp_race(model, lp, nullptr, options);
    CHECK(!r.winner.empty());
    CHECK(r.status == SimplexStatus::kOptimal);
    CHECK_NEAR(r.objective, expected_objective, 1e-5);
    CHECK(r.row_violation <= 1e-6);
    CHECK_EQ(r.engines.size(), std::size_t{3});
    std::printf("     race winner: %s (objective %.6f)\n", r.winner.c_str(), r.objective);
  };

  {
    Model m = empty_model(2);
    m.objective = {-1.0, -2.0};
    m.col_upper = {3.0, 3.0};
    std::vector<Triplet> entries;
    add_row(&m, &entries, 0, -kInf, 4.0, {{0, 1.0}, {1, 1.0}});
    m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
    run(m, -7.0);
  }
  {
    const Model m = random_lp(3, 30, 40);
    const StandardLp lp = standard(m);
    const SimplexResult cold = solve_lp(lp);
    CHECK(cold.status == SimplexStatus::kOptimal);
    run(m, cold.objective);
  }
}

// Only the engines asked for run, and the losers really do stop: with a
// single engine selected the race is just that engine, so this is really a
// check that use_ipm/use_pdhg/use_simplex are honoured and that the
// cancellation flag does not somehow stop the one engine left standing.
void test_race_can_select_one_engine() {
  Model m = empty_model(2);
  m.objective = {0.0, -1.0};
  m.col_upper = {10.0, 3.0};
  std::vector<Triplet> entries;
  add_row(&m, &entries, 0, 1.0, 5.0, {{0, 1.0}, {1, 1.0}});
  m.constraints = anukulan::SparseMatrix::from_triplets(1, 2, std::move(entries));
  const StandardLp lp = standard(m);

  for (const std::string& only : {"simplex", "ipm", "pdhg"}) {
    RaceOptions options;
    options.time_limit_seconds = 5.0;
    options.use_simplex = only == "simplex";
    options.use_ipm = only == "ipm";
    options.use_pdhg = only == "pdhg";
    const RaceResult r = solve_lp_race(m, lp, nullptr, options);
    CHECK_STR_EQ(r.winner, only);
    CHECK_NEAR(r.objective, -3.0, 1e-5);
    CHECK_EQ(r.engines.size(), std::size_t{1});
  }
}

}  // namespace

int main() {
  test_boxed_variables();
  test_free_variable();
  test_fixed_column();
  test_ranged_row();
  test_degenerate_optimal_face();
  test_primal_infeasibility_detected();
  test_dual_infeasibility_detected();
  test_crossover_from_ipm_matches_cold_simplex();
  test_race_returns_correct_answer();
  test_race_can_select_one_engine();
  return anukulan_test::finish("ipm");
}
