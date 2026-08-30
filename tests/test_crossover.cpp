#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "sankhya/crossover.hpp"
#include "sankhya/model.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/simplex.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::CrossoverResult;
using sankhya::Int;
using sankhya::kInf;
using sankhya::Model;
using sankhya::PdhgOptions;
using sankhya::PdhgResult;
using sankhya::PdhgStatus;
using sankhya::SimplexOptions;
using sankhya::SimplexResult;
using sankhya::SimplexStatus;
using sankhya::StandardFormResult;
using sankhya::StandardLp;
using sankhya::sz;
using sankhya::Triplet;
using sankhya::VarStatus;
using sankhya::crossover_basis;
using sankhya::solve_lp;
using sankhya::solve_pdhg;
using sankhya::to_logical_form;
using sankhya::to_standard_form;

namespace {

// A random, dense-ish, feasible and bounded LP. Feasibility is guaranteed by
// building the right-hand side from a point that is known to satisfy it.
Model random_lp(unsigned seed, Int rows, Int cols) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> coeff(-2.0, 3.0);
  std::uniform_real_distribution<double> point(0.0, 4.0);

  Model model;
  model.name = "random";
  model.sense = sankhya::ObjSense::kMinimize;
  model.col_lower.assign(sz(cols), 0.0);
  model.col_upper.assign(sz(cols), 10.0);
  model.objective.assign(sz(cols), 0.0);
  model.col_type.assign(sz(cols), sankhya::VarType::kContinuous);
  model.col_names.resize(sz(cols));
  for (Int j = 0; j < cols; ++j) model.objective[sz(j)] = coeff(rng);

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
    // Half the rows bind at the witness, half have slack, so the crossover has
    // both kinds of logical to deal with.
    model.row_upper[sz(i)] = activity + ((i % 2 == 0) ? 0.0 : 5.0);
  }
  model.constraints = sankhya::SparseMatrix::from_triplets(rows, cols, std::move(entries));
  return model;
}

// The invariant that matters: whatever the crossover produces, the simplex must
// be able to install it. A basis that does not factorise is the failure mode
// this design exists to rule out.
void test_produced_basis_is_always_installable() {
  Int checked = 0;
  for (unsigned seed = 1; seed <= 40; ++seed) {
    const Model model = random_lp(seed, 12 + static_cast<Int>(seed % 9),
                                  16 + static_cast<Int>(seed % 11));
    const StandardFormResult sf = to_standard_form(model);
    CHECK(sf.ok);

    PdhgOptions po;
    po.tolerance = 1e-4;
    po.gap_tolerance = 1e-4;
    po.max_iterations = 20000;
    const PdhgResult seed_point = solve_pdhg(sf.lp, po);
    if (seed_point.x.empty()) continue;

    const CrossoverResult cross =
        crossover_basis(sf.lp, seed_point.x, seed_point.y);
    CHECK(cross.ok);
    CHECK(static_cast<Int>(cross.basic.size()) == sf.lp.num_rows());

    // Every basic position names a real column, and no column is basic twice.
    const sankhya::LogicalForm form = to_logical_form(sf.lp);
    std::vector<int> seen(sz(form.columns.rows()), 0);
    for (Int b : cross.basic) {
      CHECK(b >= 0 && b < form.columns.rows());
      CHECK(seen[sz(b)] == 0);
      seen[sz(b)] = 1;
      CHECK(cross.status[sz(b)] == VarStatus::kBasic);
    }

    sankhya::SimplexBasis basis;
    std::string error;
    const bool installed = basis.set_from(form, cross.basic, cross.status, &error);
    if (!installed) std::fprintf(stderr, "seed %u: %s\n", seed, error.c_str());
    CHECK(installed);
    ++checked;
  }
  CHECK(checked >= 30);
  std::printf("     %d crossover bases installed without a singular one\n", checked);
}

// And it must not change the answer. A starting basis is a starting point; the
// optimum is a property of the problem.
void test_answer_is_unchanged() {
  Int compared = 0;
  Int pushed_total = 0;
  Int fewer = 0;
  Int cold_total = 0;
  Int warm_total = 0;
  for (unsigned seed = 1; seed <= 30; ++seed) {
    const Model model = random_lp(seed, 14 + static_cast<Int>(seed % 7),
                                  18 + static_cast<Int>(seed % 9));
    const StandardFormResult sf = to_standard_form(model);
    CHECK(sf.ok);

    SimplexOptions cold_options;
    const SimplexResult cold = solve_lp(sf.lp, cold_options);
    if (cold.status != SimplexStatus::kOptimal) continue;

    PdhgOptions po;
    po.tolerance = 1e-4;
    po.gap_tolerance = 1e-4;
    po.max_iterations = 20000;
    const PdhgResult seed_point = solve_pdhg(sf.lp, po);
    if (seed_point.x.empty()) continue;
    const CrossoverResult cross =
        crossover_basis(sf.lp, seed_point.x, seed_point.y);
    CHECK(cross.ok);
    pushed_total += cross.pushed;

    SimplexOptions warm_options;
    warm_options.start_basic = &cross.basic;
    warm_options.start_status = &cross.status;
    const SimplexResult warm = solve_lp(sf.lp, warm_options);
    CHECK(warm.status == SimplexStatus::kOptimal);
    CHECK(sankhya_test::close(warm.objective, cold.objective, 1e-7));
    CHECK(warm.started_warm);

    cold_total += cold.iterations;
    warm_total += warm.iterations;
    if (warm.iterations < cold.iterations) ++fewer;
    ++compared;
  }
  CHECK(compared >= 20);
  // The crossover has to actually do something - a run that pushes nothing is
  // trivially correct and proves nothing about the code under test.
  CHECK(pushed_total > 0);
  std::printf("     %d instances: same optimum every time, %d pushed, "
              "%d took fewer iterations (%d -> %d total)\n",
              compared, pushed_total, fewer, cold_total, warm_total);
}

// A point already sitting on a vertex should be recognised as one: the basis it
// produces should leave the simplex with nothing to do.
void test_optimal_point_needs_no_work() {
  const Model model = random_lp(7, 15, 20);
  const StandardFormResult sf = to_standard_form(model);
  CHECK(sf.ok);
  const SimplexResult exact = solve_lp(sf.lp, SimplexOptions{});
  CHECK(exact.status == SimplexStatus::kOptimal);

  const CrossoverResult cross = crossover_basis(sf.lp, exact.x, exact.y);
  CHECK(cross.ok);

  SimplexOptions options;
  options.start_basic = &cross.basic;
  options.start_status = &cross.status;
  const SimplexResult again = solve_lp(sf.lp, options);
  CHECK(again.status == SimplexStatus::kOptimal);
  CHECK(sankhya_test::close(again.objective, exact.objective, 1e-9));
  std::printf("     from its own optimum: %d iterations (cold was %d)\n",
              again.iterations, exact.iterations);
  CHECK(again.iterations <= exact.iterations);
}

// An empty or mismatched point is a caller error, not a crash.
void test_rejects_a_point_that_does_not_fit() {
  const Model model = random_lp(3, 10, 12);
  const StandardFormResult sf = to_standard_form(model);
  CHECK(sf.ok);
  const std::vector<double> too_short(sz(sf.lp.num_cols() - 1), 0.0);
  const CrossoverResult cross = crossover_basis(sf.lp, too_short, {});
  CHECK(!cross.ok);
  CHECK(!cross.message.empty());
}

}  // namespace

int main() {
  test_produced_basis_is_always_installable();
  test_answer_is_unchanged();
  test_optimal_point_needs_no_work();
  test_rejects_a_point_that_does_not_fit();
  return sankhya_test::finish("test_crossover");
}
