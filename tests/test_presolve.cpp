// Presolve is the reduction most able to return a confidently wrong answer: it
// deletes parts of the model and then claims a solution for the parts it
// deleted. So the tests here are not only "did the reduction fire" but "does
// the point that comes back out satisfy the model that went in".
#include "sankhya/presolve.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "check.hpp"
#include "sankhya/simplex.hpp"
#include "sankhya/standard_form.hpp"

using namespace sankhya;

namespace {

Model make_model(Int rows, Int cols, std::vector<Triplet> entries,
                 std::vector<double> obj, std::vector<double> rlo,
                 std::vector<double> rup, std::vector<double> clo,
                 std::vector<double> cup) {
  Model m;
  m.name = "test";
  m.constraints = SparseMatrix::from_triplets(rows, cols, std::move(entries));
  m.objective = std::move(obj);
  m.row_lower = std::move(rlo);
  m.row_upper = std::move(rup);
  m.col_lower = std::move(clo);
  m.col_upper = std::move(cup);
  m.col_type.assign(sz(cols), VarType::kContinuous);
  return m;
}

double model_objective(const Model& m, const std::vector<double>& x) {
  double v = m.objective_offset;
  for (Int j = 0; j < m.num_cols(); ++j) v += m.objective[sz(j)] * x[sz(j)];
  return v;
}

// Feasibility measured against the original model, not the reduced one.
double worst_violation(const Model& m, const std::vector<double>& x) {
  double worst = 0.0;
  for (Int j = 0; j < m.num_cols(); ++j) {
    worst = std::fmax(worst, m.col_lower[sz(j)] - x[sz(j)]);
    worst = std::fmax(worst, x[sz(j)] - m.col_upper[sz(j)]);
  }
  std::vector<double> ax(sz(m.num_rows()), 0.0);
  m.constraints.multiply(x.data(), ax.data());
  for (Int i = 0; i < m.num_rows(); ++i) {
    const double scale = std::fmax(1.0, std::fabs(ax[sz(i)]));
    worst = std::fmax(worst, (m.row_lower[sz(i)] - ax[sz(i)]) / scale);
    worst = std::fmax(worst, (ax[sz(i)] - m.row_upper[sz(i)]) / scale);
  }
  return worst;
}

bool solve_model(const Model& m, std::vector<double>* x, double* objective) {
  if (m.num_cols() == 0) {
    x->clear();
    *objective = m.objective_offset;
    return true;
  }
  const StandardFormResult sf = to_standard_form(m);
  if (!sf.ok) return false;
  SimplexOptions opt;
  opt.max_iterations = 100000;
  const SimplexResult r = solve_simplex(sf.lp, opt);
  if (r.status != SimplexStatus::kOptimal) return false;
  *x = r.x;
  *objective = r.objective;
  return true;
}

void test_empty_row() {
  // One row with no entries at all. Bounds that contain zero are satisfied by
  // nothing; bounds that exclude zero are a proof of infeasibility.
  Model m = make_model(1, 1, {}, {1.0}, {-kInf}, {5.0}, {0.0}, {10.0});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.empty_rows, 1);
  CHECK_EQ(r.reduced.num_rows(), 0);

  Model bad = make_model(1, 1, {}, {1.0}, {2.0}, {5.0}, {0.0}, {10.0});
  PresolveResult rb = presolve(bad);
  CHECK(rb.status == PresolveStatus::kInfeasible);
}

void test_singleton_row_becomes_a_bound() {
  //  2x <= 10, x in [0, 100]  ->  x in [0, 5], row gone.
  Model m = make_model(1, 1, {{0, 0, 2.0}}, {-1.0}, {-kInf}, {10.0}, {0.0}, {100.0});
  // Held one step short of the end, because once the row is gone the column is
  // in no row at all and the next reduction takes it away too.
  PresolveOptions hold;
  hold.empty_columns = false;
  PresolveResult r = presolve(m, hold);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.singleton_rows, 1);
  CHECK_EQ(r.reduced.num_rows(), 0);
  CHECK_EQ(r.reduced.num_cols(), 1);
  CHECK(r.reduced.col_upper.size() == 1 && sankhya_test::close(r.reduced.col_upper[0], 5.0, 1e-7));

  // With every reduction on, nothing is left to solve and the answer comes out
  // of postsolve alone. min -x subject to 2x <= 10 is -5.
  PresolveResult all = presolve(m);
  CHECK_EQ(all.reduced.num_cols(), 0);
  const std::vector<double> full = all.postsolve.apply({});
  CHECK_EQ(static_cast<Int>(full.size()), 1);
  CHECK_NEAR(full[0], 5.0, 1e-7);
  CHECK_NEAR(model_objective(m, full), -5.0, 1e-7);
  CHECK(worst_violation(m, full) < 1e-7);
}

void test_fixed_column_moves_into_the_row_bounds() {
  //  x + y <= 10 with x pinned at 3  ->  y <= 7, and x comes back as 3.
  Model m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {0.0, -1.0}, {-kInf},
                       {10.0}, {3.0, 0.0}, {3.0, kInf});
  PresolveOptions hold;
  hold.empty_columns = false;
  PresolveResult r = presolve(m, hold);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.fixed_columns, 1);
  CHECK_EQ(r.reduced.num_cols(), 1);
  CHECK_EQ(r.reduced.num_rows(), 0);  // x + y <= 10 became y <= 7, a bound

  std::vector<double> rx;
  double obj = 0.0;
  CHECK(solve_model(r.reduced, &rx, &obj));
  const std::vector<double> full = r.postsolve.apply(rx);
  CHECK_EQ(static_cast<Int>(full.size()), 2);
  CHECK_NEAR(full[0], 3.0, 1e-9);
  CHECK_NEAR(full[1], 7.0, 1e-7);
  CHECK_NEAR(model_objective(m, full), -7.0, 1e-7);
  CHECK(worst_violation(m, full) < 1e-7);
}

void test_empty_column_and_unboundedness() {
  //  Column 1 appears in no row. Minimising +y sends it to its lower bound.
  Model m = make_model(1, 2, {{0, 0, 1.0}}, {1.0, 2.0}, {1.0}, {kInf}, {0.0, -4.0},
                       {kInf, kInf});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  // Two, not one: y was never in a row, and x is emptied as well once the
  // singleton row that held it is folded into its bounds.
  CHECK_EQ(r.counts.empty_columns, 2);
  std::vector<double> rx;
  double obj = 0.0;
  CHECK(solve_model(r.reduced, &rx, &obj));
  const std::vector<double> full = r.postsolve.apply(rx);
  CHECK_NEAR(full[1], -4.0, 1e-9);

  // Same column with no lower bound has nothing to stop it.
  Model open = make_model(1, 2, {{0, 0, 1.0}}, {1.0, 2.0}, {1.0}, {kInf},
                          {0.0, -kInf}, {kInf, kInf});
  PresolveResult ro = presolve(open);
  CHECK(ro.status == PresolveStatus::kUnbounded);
}

void test_forcing_row_pins_everything() {
  //  x + y >= 2 with both in [0, 1]: the only way to reach 2 is both at 1.
  Model m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {2.0}, {kInf},
                       {0.0, 0.0}, {1.0, 1.0});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.forcing_rows, 1);
  CHECK_EQ(r.reduced.num_cols(), 0);
  CHECK_EQ(r.reduced.num_rows(), 0);
  const std::vector<double> full = r.postsolve.apply({});
  CHECK_NEAR(full[0], 1.0, 1e-9);
  CHECK_NEAR(full[1], 1.0, 1e-9);
  CHECK_NEAR(model_objective(m, full), 2.0, 1e-9);
}

void test_redundant_row_disappears() {
  //  x + y <= 50 cannot bind when both variables stop at 1.
  Model m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {-1.0, -1.0}, {-kInf},
                       {50.0}, {0.0, 0.0}, {1.0, 1.0});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.redundant_rows, 1);
  CHECK_EQ(r.reduced.num_rows(), 0);
}

void test_duplicate_rows_merge_into_the_tighter_one() {
  //  x + y <= 4 and 2x + 2y <= 3. The second says x + y <= 1.5, which wins.
  Model m = make_model(2, 2,
                       {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 2.0}},
                       {-1.0, -1.0}, {-kInf, -kInf}, {4.0, 3.0}, {0.0, 0.0},
                       {kInf, kInf});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.duplicate_rows, 1);
  CHECK_EQ(r.reduced.num_rows(), 1);
  CHECK_NEAR(r.reduced.row_upper[0], 1.5, 1e-9);

  std::vector<double> rx;
  double obj = 0.0;
  CHECK(solve_model(r.reduced, &rx, &obj));
  const std::vector<double> full = r.postsolve.apply(rx);
  CHECK_NEAR(model_objective(m, full), -1.5, 1e-7);
  CHECK(worst_violation(m, full) < 1e-7);
}

void test_free_column_singleton_is_substituted_out() {
  //  Row 0:  x + y      =  5      (y is free and appears nowhere else)
  //  Row 1:  x          <= 3
  //  min x + 2y  ->  y = 5 - x  ->  min 10 - x  ->  x = 3, y = 2, obj 7.
  Model m = make_model(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}},
                       {1.0, 2.0}, {5.0, -kInf}, {5.0, 3.0}, {0.0, -kInf},
                       {kInf, kInf});
  // Bound tightening off, and that is the point of the test rather than a
  // convenience. With it on, y's bounds become [2,5], y stops being free, x
  // starts looking implied-free instead, and the row dual comes back as 1 -
  // correct for the tightened model and wrong for this one, where y is free and
  // d_y = 0 forces the answer.
  PresolveOptions exact;
  exact.bound_tightening = false;
  PresolveResult r = presolve(m, exact);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.free_column_singletons, 1);
  CHECK(r.postsolve.dual_is_exact());

  std::vector<double> rx;
  double obj = 0.0;
  CHECK(solve_model(r.reduced, &rx, &obj));
  const std::vector<double> full = r.postsolve.apply(rx);
  CHECK_EQ(static_cast<Int>(full.size()), 2);
  CHECK_NEAR(full[0], 3.0, 1e-6);
  CHECK_NEAR(full[1], 2.0, 1e-6);
  CHECK_NEAR(model_objective(m, full), 7.0, 1e-6);
  CHECK(worst_violation(m, full) < 1e-7);
}

void test_maximisation_keeps_its_sign() {
  // The internal cost vector runs in the minimise direction. A maximisation
  // whose objective offset picks up a fixed column is where that bookkeeping
  // shows if it is wrong.
  Model m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {5.0, 1.0}, {-kInf},
                       {10.0}, {2.0, 0.0}, {2.0, kInf});
  m.sense = ObjSense::kMaximize;
  m.objective_offset = 100.0;
  PresolveOptions hold;
  hold.empty_columns = false;
  PresolveResult r = presolve(m, hold);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK(r.reduced.sense == ObjSense::kMaximize);
  // x is pinned at 2 and contributes 10, so the reduced offset must be 110.
  // Getting the sign wrong here would read 90 and still look plausible.
  CHECK_NEAR(r.reduced.objective_offset, 110.0, 1e-9);

  std::vector<double> rx;
  double obj = 0.0;
  CHECK(solve_model(r.reduced, &rx, &obj));
  const std::vector<double> full = r.postsolve.apply(rx);
  CHECK_NEAR(model_objective(m, full), 118.0, 1e-6);  // 100 + 5*2 + 1*8
  CHECK(worst_violation(m, full) < 1e-7);
}

void test_integer_bounds_are_rounded_not_relaxed() {
  //  3x <= 7 with x integer must give x <= 2, not x <= 2.333.
  Model m = make_model(1, 1, {{0, 0, 3.0}}, {-1.0}, {-kInf}, {7.0}, {0.0}, {100.0});
  m.col_type[0] = VarType::kInteger;
  PresolveOptions hold;
  hold.empty_columns = false;
  PresolveResult r = presolve(m, hold);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.reduced.num_cols(), 1);
  CHECK(r.reduced.col_upper.size() == 1 && sankhya_test::close(r.reduced.col_upper[0], 2.0, 1e-12));

  // A relaxed 2.333 would let branch and bound waste a node discovering what
  // the rounding already knew.
  PresolveResult all = presolve(m);
  const std::vector<double> full = all.postsolve.apply({});
  CHECK_NEAR(full[0], 2.0, 1e-12);
}

void test_hessian_blocks_column_removal() {
  // A quadratic objective makes substituting a column out a rewrite of Q as
  // well as c, which presolve does not attempt. The column must survive.
  Model m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {0.0, -1.0}, {-kInf},
                       {10.0}, {3.0, 0.0}, {3.0, kInf});
  m.hessian = SparseMatrix::from_triplets(2, 2, {{0, 0, 2.0}, {1, 1, 2.0}});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.fixed_columns, 0);
  CHECK_EQ(r.reduced.num_cols(), 2);
  CHECK_EQ(r.reduced.hessian.nnz(), 2);
}

// The real test. Random feasible bounded LPs, solved twice: straight through,
// and through presolve and back. The objectives have to agree and the restored
// point has to satisfy the original model.
void test_round_trip_against_a_direct_solve() {
  std::mt19937 rng(20260826);
  std::uniform_real_distribution<double> coeff(-3.0, 3.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int compared = 0;
  int reduced_something = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Int cols = 4 + static_cast<Int>(rng() % 8);
    const Int rows = 3 + static_cast<Int>(rng() % 7);

    std::vector<double> feasible(sz(cols));
    for (Int j = 0; j < cols; ++j) feasible[sz(j)] = unit(rng) * 5.0;

    std::vector<Triplet> entries;
    for (Int i = 0; i < rows; ++i) {
      for (Int j = 0; j < cols; ++j) {
        if (unit(rng) < 0.45) entries.push_back({i, j, coeff(rng)});
      }
    }
    Model m = make_model(rows, cols, entries, {}, {}, {}, {}, {});
    m.objective.resize(sz(cols));
    for (Int j = 0; j < cols; ++j) m.objective[sz(j)] = coeff(rng);
    m.col_lower.assign(sz(cols), 0.0);
    m.col_upper.assign(sz(cols), 10.0);

    // Row bounds built around a point that is known to be feasible, so the
    // model is never infeasible by accident. Some rows get slack big enough to
    // be redundant, some get none, some become equalities.
    std::vector<double> activity(sz(rows), 0.0);
    m.constraints.multiply(feasible.data(), activity.data());
    m.row_lower.resize(sz(rows));
    m.row_upper.resize(sz(rows));
    for (Int i = 0; i < rows; ++i) {
      const double roll = unit(rng);
      if (roll < 0.15) {
        m.row_lower[sz(i)] = activity[sz(i)];
        m.row_upper[sz(i)] = activity[sz(i)];
      } else if (roll < 0.5) {
        m.row_lower[sz(i)] = -kInf;
        m.row_upper[sz(i)] = activity[sz(i)] + unit(rng) * 40.0;
      } else if (roll < 0.8) {
        m.row_lower[sz(i)] = activity[sz(i)] - unit(rng) * 40.0;
        m.row_upper[sz(i)] = kInf;
      } else {
        m.row_lower[sz(i)] = activity[sz(i)] - unit(rng) * 2.0;
        m.row_upper[sz(i)] = activity[sz(i)] + unit(rng) * 2.0;
      }
    }

    std::vector<double> direct_x;
    double direct_obj = 0.0;
    if (!solve_model(m, &direct_x, &direct_obj)) continue;

    PresolveResult r = presolve(m);
    CHECK(r.status == PresolveStatus::kReduced);
    if (r.status != PresolveStatus::kReduced) continue;
    if (r.counts.rows_removed > 0 || r.counts.cols_removed > 0) ++reduced_something;

    std::vector<double> rx;
    double reduced_obj = 0.0;
    if (!solve_model(r.reduced, &rx, &reduced_obj)) continue;

    const std::vector<double> full = r.postsolve.apply(rx);
    CHECK_EQ(static_cast<Int>(full.size()), cols);
    CHECK_NEAR(model_objective(m, full), direct_obj, 1e-6);
    CHECK_NEAR(reduced_obj, direct_obj, 1e-6);
    const double violation = worst_violation(m, full);
    if (violation > 1e-6) {
      sankhya_test::report(__FILE__, __LINE__,
                           "restored point violates the original model by " +
                               std::to_string(violation));
    }
    ++compared;
  }
  // If the harness silently stopped solving anything, the assertions above are
  // vacuous, so the count is itself an assertion.
  CHECK(compared >= 40);
  CHECK(reduced_something >= 20);
}

// The dual side. Duals are not unique on a degenerate LP, so comparing them
// against an unpresolved solve would fail for the wrong reason. What has to
// hold is the relation each recovery rule is derived from.
void test_dual_postsolve_of_a_free_column_singleton() {
  //  Row 0:  x + y = 5     (y free, appears nowhere else, cost 2)
  //  Row 1:  x    <= 3
  //
  // y is free, so dual feasibility demands its reduced cost be zero:
  // c_y - y_0 * a_0y = 0, and with a_0y = 1 that pins y_0 = c_y = 2. There is
  // one right answer here and no degeneracy to hide behind.
  Model m = make_model(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}},
                       {1.0, 2.0}, {5.0, -kInf}, {5.0, 3.0}, {0.0, -kInf},
                       {kInf, kInf});
  // Bound tightening off, and that is the point of the test rather than a
  // convenience. With it on, y's bounds become [2,5], y stops being free, x
  // starts looking implied-free instead, and the row dual comes back as 1 -
  // correct for the tightened model and wrong for this one, where y is free and
  // d_y = 0 forces the answer.
  PresolveOptions exact;
  exact.bound_tightening = false;
  PresolveResult r = presolve(m, exact);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.free_column_singletons, 1);
  CHECK(r.postsolve.dual_is_exact());

  const std::vector<double> reduced_y(sz(r.reduced.num_rows()), 0.0);
  const std::vector<double> reduced_costs(sz(r.reduced.num_cols()), 0.0);
  const std::vector<double> y =
      r.postsolve.apply_dual(reduced_y, reduced_costs, m.objective);
  CHECK_EQ(static_cast<Int>(y.size()), 2);
  CHECK_NEAR(y[0], 2.0, 1e-12);
}

void test_rows_that_cannot_bind_get_a_zero_dual() {
  //  Row 0 is empty; row 1 cannot bind because both variables stop at one.
  Model m = make_model(2, 2, {{1, 0, 1.0}, {1, 1, 1.0}}, {-1.0, -1.0},
                       {-kInf, -kInf}, {5.0, 50.0}, {0.0, 0.0}, {1.0, 1.0});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK(r.counts.empty_rows + r.counts.redundant_rows >= 2);
  CHECK_EQ(r.postsolve.unrecoverable_rows(), 0);

  const std::vector<double> y = r.postsolve.apply_dual({}, {}, m.objective);
  CHECK_EQ(static_cast<Int>(y.size()), 2);
  CHECK_NEAR(y[0], 0.0, 1e-14);
  CHECK_NEAR(y[1], 0.0, 1e-14);
}

void test_forcing_and_duplicate_rows_are_reported_not_guessed() {
  //  x + y >= 2 with both in [0,1] forces both to one. The row's dual is only
  //  bounded by the columns' reduced costs, so there is nothing exact to put
  //  back - and the count says so rather than the value pretending.
  Model m = make_model(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, {1.0, 1.0}, {2.0},
                       {kInf}, {0.0, 0.0}, {1.0, 1.0});
  PresolveResult r = presolve(m);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.counts.forcing_rows, 1);
  CHECK_EQ(r.postsolve.unrecoverable_rows(), 1);
  CHECK(!r.postsolve.dual_is_exact());
  const std::vector<double> y = r.postsolve.apply_dual({}, {}, m.objective);
  CHECK_EQ(static_cast<Int>(y.size()), 1);
  CHECK_NEAR(y[0], 0.0, 1e-14);
}

void test_every_original_row_gets_a_dual() {
  // Whatever presolve did, the vector that comes back has to be the shape of
  // the model that went in. A silently short vector would be read as zeros.
  std::mt19937 rng(20260827);
  std::uniform_real_distribution<double> coeff(-3.0, 3.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int checked = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Int cols = 4 + static_cast<Int>(rng() % 8);
    const Int rows = 3 + static_cast<Int>(rng() % 7);
    std::vector<double> feasible(sz(cols));
    for (Int j = 0; j < cols; ++j) feasible[sz(j)] = unit(rng) * 5.0;
    std::vector<Triplet> entries;
    for (Int i = 0; i < rows; ++i)
      for (Int j = 0; j < cols; ++j)
        if (unit(rng) < 0.45) entries.push_back({i, j, coeff(rng)});
    Model m = make_model(rows, cols, entries, {}, {}, {}, {}, {});
    m.objective.resize(sz(cols));
    for (Int j = 0; j < cols; ++j) m.objective[sz(j)] = coeff(rng);
    m.col_lower.assign(sz(cols), 0.0);
    m.col_upper.assign(sz(cols), 10.0);
    std::vector<double> activity(sz(rows), 0.0);
    m.constraints.multiply(feasible.data(), activity.data());
    m.row_lower.resize(sz(rows));
    m.row_upper.resize(sz(rows));
    for (Int i = 0; i < rows; ++i) {
      m.row_lower[sz(i)] = -kInf;
      m.row_upper[sz(i)] = activity[sz(i)] + unit(rng) * 20.0;
    }
    PresolveResult r = presolve(m);
    if (r.status != PresolveStatus::kReduced) continue;
    const std::vector<double> reduced_y(sz(r.reduced.num_rows()), 0.5);
    const std::vector<double> reduced_costs(sz(r.reduced.num_cols()), 0.25);
    const std::vector<double> y =
        r.postsolve.apply_dual(reduced_y, reduced_costs, m.objective);
    CHECK_EQ(static_cast<Int>(y.size()), rows);
    for (const double v : y) CHECK(std::isfinite(v));
    ++checked;
  }
  CHECK(checked >= 30);
}


void test_dual_postsolve_indexes_reduced_costs_by_reduced_column() {
  //  x0 is fixed and disappears, so every column after it shifts down by one.
  //  Row 0 is a singleton on x2 and is removed as a bound, and recovering its
  //  dual means reading x2's reduced cost - from the reduced model, where x2 is
  //  column 1, not column 2.
  //
  //  The existing dual tests all pass an all-zero reduced-cost vector, which is
  //  the one input that cannot tell the two indexings apart.
  //  Row 0:  2 x2 <= 10          singleton, becomes a bound and is removed
  //  Row 1:  x0 + x1 + x2 = 6     equality, survives
  //  Row 2:       x1 + 2 x2 = 8   equality, survives
  Model m = make_model(3, 3,
                       {{0, 2, 2.0},
                        {1, 0, 1.0}, {1, 1, 1.0}, {1, 2, 1.0},
                        {2, 1, 1.0}, {2, 2, 2.0}},
                       {0.0, 1.0, 1.0}, {-kInf, 6.0, 8.0}, {10.0, 6.0, 8.0},
                       {1.0, 0.0, 0.0}, {1.0, kInf, kInf});
  PresolveOptions exact;
  exact.bound_tightening = false;
  PresolveResult r = presolve(m, exact);
  CHECK(r.status == PresolveStatus::kReduced);
  CHECK_EQ(r.reduced.num_cols(), 2);   // x0 is gone
  CHECK_EQ(r.counts.singleton_rows, 1);

  // Distinct per column, so reading the wrong one cannot accidentally be right.
  const std::vector<double> reduced_y(sz(r.reduced.num_rows()), 0.0);
  const std::vector<double> reduced_costs = {7.0, 3.0};  // x1 then x2
  const std::vector<double> y =
      r.postsolve.apply_dual(reduced_y, reduced_costs, m.objective);
  CHECK_EQ(static_cast<Int>(y.size()), 3);
  CHECK_NEAR(y[0], 1.5, 1e-12);  // d_x2 / a = 3.0 / 2.0
}

}  // namespace

int main() {
  test_dual_postsolve_of_a_free_column_singleton();
  test_rows_that_cannot_bind_get_a_zero_dual();
  test_forcing_and_duplicate_rows_are_reported_not_guessed();
  test_every_original_row_gets_a_dual();
  test_dual_postsolve_indexes_reduced_costs_by_reduced_column();
  test_empty_row();
  test_singleton_row_becomes_a_bound();
  test_fixed_column_moves_into_the_row_bounds();
  test_empty_column_and_unboundedness();
  test_forcing_row_pins_everything();
  test_redundant_row_disappears();
  test_duplicate_rows_merge_into_the_tighter_one();
  test_free_column_singleton_is_substituted_out();
  test_maximisation_keeps_its_sign();
  test_integer_bounds_are_rounded_not_relaxed();
  test_hessian_blocks_column_removal();
  test_round_trip_against_a_direct_solve();
  return sankhya_test::finish("presolve");
}
