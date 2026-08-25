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
  // One slack per inequality row, none for the equality.
  CHECK_EQ(form.num_rows, 4);
  CHECK_EQ(form.num_equalities, 1);
  CHECK_EQ(form.columns.rows(), form.num_structural + 3);
  // Slacks are non-negative and unbounded above.
  for (Int j = form.num_structural; j < form.columns.rows(); ++j) {
    CHECK_NEAR(form.lower[sz(j)], 0.0, 0.0);
    CHECK_NEAR(form.upper[sz(j)], kInf, 0.0);
    CHECK_NEAR(form.cost[sz(j)], 0.0, 0.0);
  }
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

int main() {
  test_logical_form_shape();
  test_basic_solution_satisfies_the_equalities();
  test_reduced_costs_vanish_on_the_basis();
  test_pivoting_preserves_the_invariant();
  test_larger_instance_factorizes_and_solves();
  return sankhya_test::finish("test_simplex");
}
