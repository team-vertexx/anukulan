#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "check.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/simplex.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::Int;
using sankhya::kInf;
using sankhya::LogicalForm;
using sankhya::SimplexBasis;
using sankhya::StandardFormResult;
using sankhya::sz;
using sankhya::Model;
using sankhya::SimplexOptions;
using sankhya::SimplexResult;
using sankhya::SimplexStatus;
using sankhya::SparseMatrix;
using sankhya::Triplet;
using sankhya::VarType;
using sankhya::solve_dual_simplex;
using sankhya::solve_lp;
using sankhya::solve_simplex;
using sankhya::to_standard_form;
using sankhya::to_string;
using sankhya::VarStatus;

namespace {

LogicalForm build(const std::string& text) {
  std::istringstream in(text);
  const sankhya::MpsReadResult r = sankhya::read_mps_stream(in, "<test>");
  if (!r.ok) std::fprintf(stderr, "read failed: %s\n", r.error.c_str());
  CHECK(r.ok);
  const StandardFormResult sf = sankhya::to_standard_form(r.model);
  CHECK(sf.ok);
  return sankhya::to_logical_form(sf.lp);
}

const char* kMixed =
    "NAME          MIX\n"
    "ROWS\n"
    " N  COST\n"
    " E  EQ1\n"
    " L  LE1\n"
    " G  GE1\n"
    " L  LE2\n"
    "COLUMNS\n"
    "    X         COST      2.0        EQ1       1.0\n"
    "    X         LE1       3.0        GE1       1.0\n"
    "    Y         COST      3.0        EQ1       2.0\n"
    "    Y         LE1       1.0        LE2       4.0\n"
    "    Z         COST      -1.0       GE1       2.0\n"
    "    Z         LE2       1.0        LE1       1.0\n"
    "RHS\n"
    "    RHS       EQ1       6.0        LE1       20.0\n"
    "    RHS       GE1       2.0        LE2       18.0\n"
    "BOUNDS\n"
    " UP BND       X         8.0\n"
    " UP BND       Y         9.0\n"
    " UP BND       Z         7.0\n"
    "ENDATA\n";

// A z, computed from the column store.
void activity(const LogicalForm& form, const std::vector<double>& z,
              std::vector<double>* out) {
  out->assign(sz(form.num_rows), 0.0);
  for (Int j = 0; j < form.columns.rows(); ++j) {
    const double value = z[sz(j)];
    if (value == 0.0) continue;
    for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
      (*out)[sz(form.columns.index()[sz(e)])] += form.columns.value()[sz(e)] * value;
    }
  }
}

void test_logical_form_shape() {
  const LogicalForm form = build(kMixed);
  // Every row gets a logical, equalities included. The equality's logical is
  // fixed at zero, so it changes nothing about the feasible set, and in return
  // the all-logical basis is the identity and cannot be singular.
  CHECK_EQ(form.num_rows, 4);
  CHECK_EQ(form.num_equalities, 1);
  CHECK_EQ(form.columns.rows(), form.num_structural + form.num_rows);

  for (Int j = form.num_structural; j < form.columns.rows(); ++j) {
    const Int row = j - form.num_structural;
    CHECK_NEAR(form.lower[sz(j)], 0.0, 0.0);
    CHECK_NEAR(form.cost[sz(j)], 0.0, 0.0);
    if (row < form.num_equalities) {
      CHECK_NEAR(form.upper[sz(j)], 0.0, 0.0);  // pinned shut
    } else {
      CHECK_NEAR(form.upper[sz(j)], kInf, 0.0);
    }
  }
}

void test_initial_basis_is_the_identity() {
  // Whatever the model, the starting basis is the logical one, so the first
  // factorisation is of -I and always succeeds. An earlier version picked a
  // structural column for each equality row by a greedy rule and produced
  // singular bases on scfxm1, bandm and degen2 - all three failed before the
  // first iteration.
  const LogicalForm form = build(kMixed);
  SimplexBasis basis;
  std::string error;
  CHECK(basis.set_initial(form, &error));
  for (Int i = 0; i < form.num_rows; ++i) {
    CHECK_EQ(basis.basic()[sz(i)], form.num_structural + i);
  }
  CHECK_EQ(basis.factors().nucleus_size(), 0);
}

void test_basic_solution_satisfies_the_equalities() {
  // The one invariant the whole method rests on: whatever the basis, the point
  // it implies must satisfy A z = q exactly. If it does not, every ratio test
  // and every reduced cost afterwards is computed about a point that is not on
  // the feasible set at all.
  const LogicalForm form = build(kMixed);
  SimplexBasis basis;
  std::string error;
  CHECK(basis.set_initial(form, &error));
  if (!error.empty()) std::fprintf(stderr, "initial basis: %s\n", error.c_str());

  std::vector<double> z;
  basis.compute_primal(form, &z);
  std::vector<double> az;
  activity(form, z, &az);
  double worst = 0.0;
  for (Int i = 0; i < form.num_rows; ++i) {
    worst = std::fmax(worst, std::fabs(az[sz(i)] - form.rhs[sz(i)]));
  }
  std::printf("     initial basis: worst equality violation %.2e\n", worst);
  CHECK(worst < 1e-11);
}

void test_reduced_costs_vanish_on_the_basis() {
  // By construction the basic reduced costs are zero. Computing them a
  // different way and finding something else means the duals are wrong.
  const LogicalForm form = build(kMixed);
  SimplexBasis basis;
  CHECK(basis.set_initial(form));

  std::vector<double> duals;
  std::vector<double> reduced;
  basis.compute_duals(form, &duals, &reduced);

  double worst = 0.0;
  for (Int i = 0; i < form.num_rows; ++i) {
    const Int j = basis.basic()[sz(i)];
    double dot = 0.0;
    for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
      dot += form.columns.value()[sz(e)] * duals[sz(form.columns.index()[sz(e)])];
    }
    worst = std::fmax(worst, std::fabs(form.cost[sz(j)] - dot));
  }
  std::printf("     duals: worst basic reduced cost %.2e\n", worst);
  CHECK(worst < 1e-11);
}

void test_pivoting_preserves_the_invariant() {
  // Pivot repeatedly and check after each one that the basis is still valid and
  // still reproduces the right-hand side. Random pivots reach bases the
  // algorithm itself might take many iterations to find.
  const LogicalForm form = build(kMixed);
  SimplexBasis basis;
  CHECK(basis.set_initial(form));

  std::mt19937 rng(20260825);
  std::vector<double> z;
  std::vector<double> az;
  std::vector<double> column;
  int pivots = 0;
  double worst = 0.0;

  for (int attempt = 0; attempt < 200 && pivots < 60; ++attempt) {
    const Int row = static_cast<Int>(rng() % static_cast<unsigned>(form.num_rows));
    const Int entering =
        static_cast<Int>(rng() % static_cast<unsigned>(form.columns.rows()));
    if (basis.status()[sz(entering)] == VarStatus::kBasic) continue;

    // Only pivot on an entry large enough to be a sane pivot, which is what the
    // ratio test would enforce.
    basis.ftran_column(form, entering, &column);
    if (std::fabs(column[sz(row)]) < 1e-6) continue;

    const VarStatus leaving_to =
        std::isfinite(form.lower[sz(basis.basic()[sz(row)])]) ? VarStatus::kAtLower
                                                              : VarStatus::kAtUpper;
    if (!basis.pivot(form, row, entering, leaving_to)) continue;
    ++pivots;

    basis.compute_primal(form, &z);
    activity(form, z, &az);
    for (Int i = 0; i < form.num_rows; ++i) {
      worst = std::fmax(worst, std::fabs(az[sz(i)] - form.rhs[sz(i)]));
    }
  }
  std::printf("     %d random pivots: worst equality violation %.2e\n", pivots, worst);
  CHECK(pivots > 20);
  CHECK(worst < 1e-9);
}

void test_larger_instance_factorizes_and_solves() {
  // A wider model, to check the initial basis construction copes when equality
  // rows outnumber convenient columns.
  std::string text =
      "NAME          WIDE\n"
      "ROWS\n"
      " N  COST\n";
  for (int i = 0; i < 12; ++i) {
    text += std::string(" ") + (i % 3 == 0 ? "E" : (i % 3 == 1 ? "L" : "G")) +
            "  R" + std::to_string(i) + "\n";
  }
  text += "COLUMNS\n";
  for (int j = 0; j < 20; ++j) {
    const std::string col = "C" + std::to_string(j);
    text += "    " + col + "        COST      " + std::to_string(1.0 + j % 5) + "\n";
    for (int k = 0; k < 3; ++k) {
      const int row = (j * 5 + k * 4) % 12;
      text += "    " + col + "        R" + std::to_string(row) + "        " +
              std::to_string(1.0 + ((j + k) % 4)) + "\n";
    }
  }
  text += "RHS\n";
  for (int i = 0; i < 12; ++i) {
    text += "    RHS       R" + std::to_string(i) + "        " +
            std::to_string(5.0 + i) + "\n";
  }
  text += "BOUNDS\n";
  for (int j = 0; j < 20; ++j) {
    text += " UP BND       C" + std::to_string(j) + "        30.0\n";
  }
  text += "ENDATA\n";

  const LogicalForm form = build(text);
  SimplexBasis basis;
  std::string error;
  CHECK(basis.set_initial(form, &error));
  if (!error.empty()) std::fprintf(stderr, "wide basis: %s\n", error.c_str());

  std::vector<double> z;
  basis.compute_primal(form, &z);
  std::vector<double> az;
  activity(form, z, &az);
  double worst = 0.0;
  for (Int i = 0; i < form.num_rows; ++i) {
    worst = std::fmax(worst, std::fabs(az[sz(i)] - form.rhs[sz(i)]));
  }
  std::printf("     wide model %dx%d: worst equality violation %.2e\n",
              form.num_rows, form.columns.rows(), worst);
  CHECK(worst < 1e-10);
}

}  // namespace

// The dual simplex has to end up where the primal does. Random boxed LPs are
// used because every column having two finite bounds is exactly the case the
// dual can always start from: a boxed variable is dual feasible on whichever
// bound matches the sign of its reduced cost, so no phase one is needed and the
// solver is actually exercised rather than falling back.
void test_dual_matches_primal() {
  std::mt19937 rng(20260826);
  std::uniform_real_distribution<double> coeff(-3.0, 3.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int compared = 0;
  int ran_dual = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Int cols = 4 + static_cast<Int>(rng() % 8);
    const Int rows = 3 + static_cast<Int>(rng() % 6);

    std::vector<double> feasible(sz(cols));
    for (Int j = 0; j < cols; ++j) feasible[sz(j)] = unit(rng) * 4.0;

    std::vector<Triplet> entries;
    for (Int i = 0; i < rows; ++i)
      for (Int j = 0; j < cols; ++j)
        if (unit(rng) < 0.5) entries.push_back({i, j, coeff(rng)});

    Model m;
    m.name = "dual";
    m.constraints = SparseMatrix::from_triplets(rows, cols, entries);
    m.objective.resize(sz(cols));
    for (Int j = 0; j < cols; ++j) m.objective[sz(j)] = coeff(rng);
    m.col_lower.assign(sz(cols), 0.0);
    m.col_upper.assign(sz(cols), 8.0);
    m.col_type.assign(sz(cols), VarType::kContinuous);

    std::vector<double> activity(sz(rows), 0.0);
    m.constraints.multiply(feasible.data(), activity.data());
    m.row_lower.resize(sz(rows));
    m.row_upper.resize(sz(rows));
    for (Int i = 0; i < rows; ++i) {
      if (unit(rng) < 0.3) {
        m.row_lower[sz(i)] = activity[sz(i)];
        m.row_upper[sz(i)] = activity[sz(i)];
      } else {
        m.row_lower[sz(i)] = -kInf;
        m.row_upper[sz(i)] = activity[sz(i)] + unit(rng) * 10.0;
      }
    }

    const StandardFormResult sf = to_standard_form(m);
    if (!sf.ok) continue;

    SimplexOptions primal_options;
    primal_options.max_iterations = 50000;
    SimplexOptions dual_options = primal_options;
    dual_options.algorithm = SimplexOptions::Algorithm::kDual;

    const SimplexResult p = solve_simplex(sf.lp, primal_options);
    const SimplexResult d = solve_lp(sf.lp, dual_options);
    if (p.status != SimplexStatus::kOptimal) continue;
    if (d.status != SimplexStatus::kOptimal) {
      sankhya_test::report(__FILE__, __LINE__,
                           "dual did not reach optimal where the primal did: " +
                               to_string(d.status));
      continue;
    }
    if (!d.fell_back_to_primal) ++ran_dual;
    CHECK_NEAR(d.objective, p.objective, 1e-7);
    ++compared;
  }
  // Vacuous assertions are the failure mode here, so the counts are assertions
  // too: the dual has to have actually run, not merely handed back every time.
  CHECK(compared >= 25);
  CHECK(ran_dual >= 20);
  std::printf("     dual vs primal: %d compared, %d actually ran the dual\n",
              compared, ran_dual);
}

int main() {
  test_dual_matches_primal();
  test_logical_form_shape();
  test_initial_basis_is_the_identity();
  test_basic_solution_satisfies_the_equalities();
  test_reduced_costs_vanish_on_the_basis();
  test_pivoting_preserves_the_invariant();
  test_larger_instance_factorizes_and_solves();
  return sankhya_test::finish("test_simplex");
}
