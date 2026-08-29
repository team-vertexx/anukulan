#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "check.hpp"
#include "sankhya/cuts.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/simplex.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::Cut;
using sankhya::CutOptions;
using sankhya::Int;
using sankhya::kInf;
using sankhya::Model;
using sankhya::StandardFormResult;
using sankhya::StandardLp;
using sankhya::sz;

namespace {

// The one property that matters. A cut that removes a feasible integer point
// silently changes the answer, and nothing downstream can detect it - the solver
// will report a confident optimum for a problem nobody posed.
//
// For small enough models every feasible integer point can simply be
// enumerated, so validity is checked by exhaustion rather than by argument.

StandardFormResult build(const std::string& text) {
  std::istringstream in(text);
  const sankhya::MpsReadResult r = sankhya::read_mps_stream(in, "<test>");
  if (!r.ok) std::fprintf(stderr, "read failed: %s\n", r.error.c_str());
  CHECK(r.ok);
  return sankhya::to_standard_form(r.model);
}

std::vector<bool> integrality(const Model& m) {
  std::vector<bool> flags(sz(m.num_cols()), false);
  for (Int j = 0; j < m.num_cols(); ++j)
    flags[sz(j)] = m.col_type[sz(j)] != sankhya::VarType::kContinuous;
  return flags;
}

// Every integer point inside the bounds, filtered to those that satisfy the
// constraints.
void enumerate(const StandardLp& lp, const std::vector<bool>& integral,
               std::vector<std::vector<double>>* out, std::vector<double> partial,
               std::size_t index) {
  if (index == sz(lp.num_cols())) {
    std::vector<double> scratch;
    double inf_norm = 0.0;
    lp.primal_residual(partial, &scratch, nullptr, &inf_norm);
    if (inf_norm <= 1e-9) out->push_back(partial);
    return;
  }
  CHECK(integral[index]);  // the fixtures here are pure integer programs
  const double lo = lp.lower[index];
  const double hi = lp.upper[index];
  CHECK(lo > -kInf && hi < kInf);
  for (double v = std::ceil(lo - 1e-9); v <= hi + 1e-9; v += 1.0) {
    partial[index] = v;
    enumerate(lp, integral, out, partial, index + 1);
  }
}

void check_gomory_cuts_are_valid(const std::string& text, const char* label);

void check_all_cuts_are_valid(const std::string& text, const char* label) {
  const StandardFormResult sf = build(text);
  CHECK(sf.ok);
  if (!sf.ok) return;
  std::istringstream in(text);
  const Model model = sankhya::read_mps_stream(in, "<test>").model;
  const std::vector<bool> integral = integrality(model);

  std::vector<std::vector<double>> feasible;
  enumerate(sf.lp, integral, &feasible,
            std::vector<double>(sz(sf.lp.num_cols()), 0.0), 0);
  if (feasible.empty()) {
    std::fprintf(stderr, "%s: no feasible integer points to check against\n", label);
    CHECK(!feasible.empty());
    return;
  }

  // Separate at many points, not just the relaxation optimum: a separator that
  // is only ever asked about one point is barely tested.
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<double> pick(0.0, 1.0);
  Int cuts_generated = 0;

  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> x(sz(sf.lp.num_cols()));
    for (Int j = 0; j < sf.lp.num_cols(); ++j) {
      const double lo = sf.lp.lower[sz(j)];
      const double hi = sf.lp.upper[sz(j)];
      x[sz(j)] = lo + pick(rng) * (hi - lo);
    }
    const std::vector<Cut> cuts = sankhya::separate_cuts(sf.lp, integral, x);
    cuts_generated += static_cast<Int>(cuts.size());

    for (const Cut& cut : cuts) {
      for (const std::vector<double>& point : feasible) {
        double activity = 0.0;
        for (std::size_t idx = 0; idx < cut.columns.size(); ++idx)
          activity += cut.coefficients[idx] * point[sz(cut.columns[idx])];
        if (activity < cut.rhs - 1e-7) {
          std::fprintf(stderr,
                       "%s: a %s cut removes a feasible integer point "
                       "(activity %.10g < rhs %.10g)\n",
                       label, cut.family, activity, cut.rhs);
          CHECK(false);
          return;
        }
      }
    }
  }
  // And at vertices, which is a different regime from a random interior point
  // and the one branch and cut actually separates at once the root is solved by
  // a simplex rather than a first-order method. A separator can be sound on the
  // inside of the box and wrong on a face.
  Int vertex_cuts = 0;
  for (int trial = 0; trial < 40; ++trial) {
    sankhya::StandardLp lp = sf.lp;
    for (Int j = 0; j < lp.num_cols(); ++j) lp.c[sz(j)] = pick(rng) * 4.0 - 2.0;
    sankhya::SimplexOptions so;
    so.max_iterations = 20000;
    const sankhya::SimplexResult r = sankhya::solve_simplex(lp, so);
    if (r.status != sankhya::SimplexStatus::kOptimal) continue;
    const std::vector<Cut> cuts = sankhya::separate_cuts(sf.lp, integral, r.x);
    vertex_cuts += static_cast<Int>(cuts.size());
    for (const Cut& cut : cuts) {
      for (const std::vector<double>& point : feasible) {
        double activity = 0.0;
        for (std::size_t idx = 0; idx < cut.columns.size(); ++idx)
          activity += cut.coefficients[idx] * point[sz(cut.columns[idx])];
        if (activity < cut.rhs - 1e-7) {
          std::fprintf(stderr,
                       "%s: a %s cut separated at a vertex removes a feasible "
                       "integer point (activity %.10g < rhs %.10g)\n",
                       label, cut.family, activity, cut.rhs);
          CHECK(false);
          return;
        }
      }
    }
  }

  std::printf("     %s: %d feasible points, %d cuts at random points, %d at "
              "vertices, none invalid\n", label,
              static_cast<int>(feasible.size()), cuts_generated,
              static_cast<int>(vertex_cuts));
  CHECK(cuts_generated > 0);  // a separator that never fires proves nothing

  // Every fixture gets put through the tableau separator too, so a new fixture
  // covers both without anyone remembering to add it twice.
  check_gomory_cuts_are_valid(text, label);
}

void test_binary_knapsack() {
  check_all_cuts_are_valid(
      "NAME          KNAP\n"
      "ROWS\n"
      " N  obj\n"
      " L  cap\n"
      "COLUMNS\n"
      "    MARKER                 'MARKER'                 'INTORG'\n"
      "    x1        obj       -16.0      cap       5.0\n"
      "    x2        obj       -22.0      cap       7.0\n"
      "    x3        obj       -12.0      cap       4.0\n"
      "    x4        obj       -8.0       cap       3.0\n"
      "    x5        obj       -11.0      cap       6.0\n"
      "    MARKER                 'MARKER'                 'INTEND'\n"
      "RHS\n"
      "    RHS       cap       14.0\n"
      "BOUNDS\n"
      " UP BND       x1        1.0\n"
      " UP BND       x2        1.0\n"
      " UP BND       x3        1.0\n"
      " UP BND       x4        1.0\n"
      " UP BND       x5        1.0\n"
      "ENDATA\n",
      "binary knapsack");
}

void test_general_integer_rows() {
  // General integers rather than binaries, so the MIR path is exercised and the
  // cover path is not.
  check_all_cuts_are_valid(
      "NAME          GENINT\n"
      "ROWS\n"
      " N  obj\n"
      " L  r1\n"
      " L  r2\n"
      " G  r3\n"
      "COLUMNS\n"
      "    MARKER                 'MARKER'                 'INTORG'\n"
      "    y1        obj       -3.0       r1        2.5\n"
      "    y1        r2        1.0        r3        1.0\n"
      "    y2        obj       -4.0       r1        3.5\n"
      "    y2        r2        -1.0       r3        1.0\n"
      "    y3        obj       -2.0       r1        1.5\n"
      "    y3        r2        2.0\n"
      "    MARKER                 'MARKER'                 'INTEND'\n"
      "RHS\n"
      "    RHS       r1        11.5       r2        6.0\n"
      "    RHS       r3        1.0\n"
      "BOUNDS\n"
      " UP BND       y1        4.0\n"
      " UP BND       y2        4.0\n"
      " UP BND       y3        4.0\n"
      "ENDATA\n",
      "general integers");
}

void test_equality_rows() {
  // Equalities are read from both directions, which doubles the chances of a
  // sign slip in the separator.
  check_all_cuts_are_valid(
      "NAME          EQROW\n"
      "ROWS\n"
      " N  obj\n"
      " E  e1\n"
      " L  r2\n"
      "COLUMNS\n"
      "    MARKER                 'MARKER'                 'INTORG'\n"
      "    z1        obj       -5.0       e1        2.0\n"
      "    z1        r2        3.5\n"
      "    z2        obj       -3.0       e1        3.0\n"
      "    z2        r2        1.5\n"
      "    z3        obj       -1.0       e1        1.0\n"
      "    z3        r2        2.5\n"
      "    MARKER                 'MARKER'                 'INTEND'\n"
      "RHS\n"
      "    RHS       e1        6.0        r2        9.5\n"
      "BOUNDS\n"
      " UP BND       z1        3.0\n"
      " UP BND       z2        3.0\n"
      " UP BND       z3        3.0\n"
      "ENDATA\n",
      "equality rows");
}

// Hand-written fixtures test the shapes someone thought of. This generates
// many small integer programs instead and checks every cut against every
// feasible point of each, which is how the c-MIR separator's equality-row bug
// was found: fiber has 363 equality rows and 1,254 binaries, and the separator
// produced a cut that removed the optimum, so the solver reported a confident
// optimal objective 60.8% away from the published one.
void test_random_integer_programs() {
  std::mt19937 rng(20260829);
  std::uniform_int_distribution<int> ncols(4, 8);
  std::uniform_int_distribution<int> nrows(1, 3);
  // fiber's coefficients run from 1 to 216, and a narrow range makes the MIR
  // scalings all look alike, so this spreads them.
  std::uniform_int_distribution<int> coef(-9, 9);
  std::uniform_int_distribution<int> big(-40, 40);
  std::uniform_int_distribution<int> use_big(0, 4);
  std::uniform_int_distribution<int> kind(0, 2);   // 0 = E, 1 = L, 2 = G
  std::uniform_int_distribution<int> rhs_pick(-6, 12);
  // Binary columns as well as general integers: fiber is 1,254 binaries, and
  // the complementation rule behaves differently on a range of one.
  std::uniform_int_distribution<int> ub_pick(0, 1);

  int generated = 0;
  Int total_cuts = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const int n = ncols(rng);
    const int m = nrows(rng);
    std::vector<std::vector<int>> a(sz(m), std::vector<int>(sz(n), 0));
    std::vector<int> b(sz(m));
    std::vector<char> rowtype(sz(m));
    bool any_nonzero = false;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        a[sz(i)][sz(j)] = (use_big(rng) == 0) ? big(rng) : coef(rng);
        if (a[sz(i)][sz(j)] != 0) any_nonzero = true;
      }
      b[sz(i)] = rhs_pick(rng);
      rowtype[sz(i)] = "ELG"[sz(kind(rng))];
    }
    if (!any_nonzero) continue;

    std::ostringstream mps;
    mps << "NAME          RAND\nROWS\n N  COST\n";
    for (int i = 0; i < m; ++i) mps << " " << rowtype[sz(i)] << "  R" << i << "\n";
    mps << "COLUMNS\n    MARKER                 'MARKER'                 'INTORG'\n";
    for (int j = 0; j < n; ++j) {
      mps << "    X" << j << "  COST  " << (coef(rng)) << ".0\n";
      for (int i = 0; i < m; ++i) {
        if (a[sz(i)][sz(j)] == 0) continue;
        mps << "    X" << j << "  R" << i << "  " << a[sz(i)][sz(j)] << ".0\n";
      }
    }
    mps << "    MARKER                 'MARKER'                 'INTEND'\nRHS\n";
    for (int i = 0; i < m; ++i) mps << "    RHS  R" << i << "  " << b[sz(i)] << ".0\n";
    mps << "BOUNDS\n";
    const int ub = ub_pick(rng) ? 1 : 3;
    for (int j = 0; j < n; ++j)
      mps << " UP BND       X" << j << "  " << ub << ".0\n";
    mps << "ENDATA\n";

    // Only usable as a fixture when it has feasible integer points at all.
    const StandardFormResult sf = build(mps.str());
    if (!sf.ok) continue;
    std::istringstream in(mps.str());
    const Model model = sankhya::read_mps_stream(in, "<rand>").model;
    std::vector<std::vector<double>> feasible;
    enumerate(sf.lp, integrality(model), &feasible,
              std::vector<double>(sz(sf.lp.num_cols()), 0.0), 0);
    if (feasible.empty()) continue;

    ++generated;

    // Separate at many points rather than only the relaxation optimum, and
    // check every cut against every feasible integer point. Most of these
    // fixtures are small enough that nothing is separated at all, which is why
    // the cut count is asserted over the whole run rather than per fixture.
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    for (int trial2 = 0; trial2 < 40; ++trial2) {
      std::vector<double> x(sz(sf.lp.num_cols()));
      for (Int j = 0; j < sf.lp.num_cols(); ++j) {
        const double lo = sf.lp.lower[sz(j)];
        const double hi = sf.lp.upper[sz(j)];
        x[sz(j)] = std::isfinite(lo) && std::isfinite(hi)
                       ? lo + pick(rng) * (hi - lo)
                       : (std::isfinite(lo) ? lo + pick(rng) * 4.0 : pick(rng));
      }
      const std::vector<Cut> cuts =
          sankhya::separate_cuts(sf.lp, integrality(model), x);
      total_cuts += static_cast<Int>(cuts.size());
      for (const Cut& cut : cuts) {
        for (const std::vector<double>& point : feasible) {
          double activity = 0.0;
          for (std::size_t idx = 0; idx < cut.columns.size(); ++idx)
            activity += cut.coefficients[idx] * point[sz(cut.columns[idx])];
          if (activity < cut.rhs - 1e-7) {
            std::fprintf(stderr,
                         "random program %d: a %s cut removes a feasible "
                         "integer point (activity %.10g < rhs %.10g)\n",
                         trial, cut.family, activity, cut.rhs);
            CHECK(activity >= cut.rhs - 1e-7);
            return;
          }
        }
      }
    }
  }
  // If the generator stopped producing usable fixtures, or stopped provoking
  // any separation, the test would pass while checking nothing.
  CHECK(generated >= 20);
  CHECK(total_cuts > 0);
}

void test_cuts_actually_tighten() {
  // Validity is necessary but useless on its own: a separator that returns
  // nothing is trivially valid. This checks the cuts do cut, by confirming the
  // relaxation bound moves when they are added.
  const std::string text =
      "NAME          TIGHT\n"
      "ROWS\n"
      " N  obj\n"
      " L  cap\n"
      "COLUMNS\n"
      "    MARKER                 'MARKER'                 'INTORG'\n"
      "    x1        obj       -10.0      cap       6.0\n"
      "    x2        obj       -10.0      cap       6.0\n"
      "    x3        obj       -10.0      cap       6.0\n"
      "    MARKER                 'MARKER'                 'INTEND'\n"
      "RHS\n"
      "    RHS       cap       14.0\n"
      "BOUNDS\n"
      " UP BND       x1        1.0\n"
      " UP BND       x2        1.0\n"
      " UP BND       x3        1.0\n"
      "ENDATA\n";
  const StandardFormResult sf = build(text);
  CHECK(sf.ok);
  if (!sf.ok) return;
  std::istringstream in(text);
  const Model model = sankhya::read_mps_stream(in, "<test>").model;

  sankhya::PdhgOptions opt;
  opt.tolerance = 1e-9;
  const sankhya::PdhgResult before = sankhya::solve_pdhg(sf.lp, opt);
  CHECK(before.status == sankhya::PdhgStatus::kOptimal);

  const std::vector<Cut> cuts =
      sankhya::separate_cuts(sf.lp, integrality(model), before.x);
  CHECK(!cuts.empty());
  if (cuts.empty()) return;

  const StandardLp tightened = sankhya::append_cuts(sf.lp, cuts);
  CHECK_EQ(tightened.num_rows(), sf.lp.num_rows() + static_cast<Int>(cuts.size()));
  const sankhya::PdhgResult after = sankhya::solve_pdhg(tightened, opt);
  CHECK(after.status == sankhya::PdhgStatus::kOptimal);

  // Two of three items fit, so the integer optimum is -20 while the relaxation
  // reaches -23.33. A valid cut moves the bound up toward -20, never past it.
  std::printf("     tightening: relaxation %.6f -> %.6f (integer optimum -20)\n",
              before.objective, after.objective);
  CHECK(after.objective > before.objective + 1e-6);
  CHECK(after.objective <= -20.0 + 1e-6);
}

// The same exhaustion test, for the cuts that come off the tableau instead of
// off the model's own rows.
//
// A Gomory cut is a combination of rows with a rounding argument on top, so a
// sign error anywhere in it - which bound a nonbasic sits on, which way a slack
// substitutes back - produces an inequality that looks perfectly sensible and
// silently removes integer solutions. Nothing short of enumerating the feasible
// points catches that, which is why this test is the gate on the whole feature.
//
// One basis yields a handful of rows worth cutting, so the LP is re-solved
// under many random objectives and cuts are taken from every basis that
// produces. Different objectives put different columns on different bounds,
// which is exactly the part most likely to be wrong.
void check_gomory_cuts_are_valid(const std::string& text, const char* label) {
  const StandardFormResult sf = build(text);
  CHECK(sf.ok);
  if (!sf.ok) return;
  std::istringstream in(text);
  const sankhya::Model model = sankhya::read_mps_stream(in, "<test>").model;
  const std::vector<bool> integral = integrality(model);

  std::vector<std::vector<double>> feasible;
  enumerate(sf.lp, integral, &feasible,
            std::vector<double>(sz(sf.lp.num_cols()), 0.0), 0);
  CHECK(!feasible.empty());
  if (feasible.empty()) return;

  std::mt19937 rng(20260826);
  std::uniform_real_distribution<double> pick(-2.0, 2.0);
  Int cuts_generated = 0;
  Int bases_used = 0;

  for (int trial = 0; trial < 60; ++trial) {
    sankhya::StandardLp lp = sf.lp;
    if (trial > 0) {
      for (Int j = 0; j < lp.num_cols(); ++j) lp.c[sz(j)] = pick(rng);
    }
    sankhya::SimplexOptions options;
    options.max_iterations = 20000;
    const sankhya::SimplexResult r = sankhya::solve_simplex(lp, options);
    if (r.status != sankhya::SimplexStatus::kOptimal) continue;
    if (r.final_basic.empty()) continue;
    ++bases_used;

    const std::vector<Cut> cuts = sankhya::separate_gomory_cuts(
        sf.lp, integral, r.final_basic, r.final_status);
    cuts_generated += static_cast<Int>(cuts.size());

    for (const Cut& cut : cuts) {
      for (const std::vector<double>& point : feasible) {
        double activity = 0.0;
        for (std::size_t idx = 0; idx < cut.columns.size(); ++idx)
          activity += cut.coefficients[idx] * point[sz(cut.columns[idx])];
        if (activity < cut.rhs - 1e-6) {
          std::fprintf(stderr,
                       "%s: a gomory cut removes a feasible integer point "
                       "(activity %.10g < rhs %.10g)\n",
                       label, activity, cut.rhs);
          CHECK(false);
          return;
        }
      }
    }
  }
  std::printf("     %s: %d feasible points, %d bases, %d gomory cuts, none invalid\n",
              label, static_cast<int>(feasible.size()),
              static_cast<int>(bases_used), static_cast<int>(cuts_generated));
  CHECK(bases_used >= 10);
  CHECK(cuts_generated > 0);  // a separator that never fires proves nothing
}

}  // namespace

int main() {
  test_binary_knapsack();
  test_general_integer_rows();
  test_equality_rows();
  test_random_integer_programs();
  test_cuts_actually_tighten();
  return sankhya_test::finish("test_cuts");
}
