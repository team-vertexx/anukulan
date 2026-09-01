#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "sankhya/branch_and_bound.hpp"
#include "sankhya/model.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/standard_form.hpp"

using sankhya::BranchAndBoundOptions;
using sankhya::BranchAndBoundResult;
using sankhya::Int;
using sankhya::kInf;
using sankhya::MilpStatus;
using sankhya::Model;
using sankhya::ObjSense;
using sankhya::sz;

namespace {

// The tree had no test of its own, which is the wrong way round: it is the
// component this repository's own assessment rates lowest and the one that has
// twice returned a proved optimum that was wrong. A slow answer announces
// itself. A wrong one does not.
//
// So the property under test here is the only one that matters. For a model
// small enough to enumerate, the answer branch and bound proves is compared
// against the answer obtained by trying every integer point. Nothing about how
// it got there is checked - a tree that prunes differently is allowed, a tree
// that prunes away the optimum is not.

Model read(const std::string& text) {
  std::istringstream in(text);
  const sankhya::MpsReadResult r = sankhya::read_mps_stream(in, "<test>");
  if (!r.ok) std::fprintf(stderr, "read failed: %s\n", r.error.c_str());
  CHECK(r.ok);
  return r.model;
}

// Every integer point inside the bounds that satisfies the rows, and the best
// objective over them in the model's own sense. Exhaustive, so the fixtures
// have to stay small - which is the point.
void enumerate(const sankhya::StandardLp& lp, const std::vector<bool>& integral,
               std::vector<double>* point, std::size_t index, bool* any,
               double* best) {
  if (index == sz(lp.num_cols())) {
    std::vector<double> scratch;
    double inf_norm = 0.0;
    lp.primal_residual(*point, &scratch, nullptr, &inf_norm);
    if (inf_norm > 1e-9) return;
    const double value = lp.standard_objective(*point);
    if (!*any || value < *best) {
      *best = value;
      *any = true;
    }
    return;
  }
  CHECK(integral[index]);  // these fixtures are pure integer programs
  const double lo = lp.lower[index];
  const double hi = lp.upper[index];
  CHECK(std::isfinite(lo) && std::isfinite(hi));
  for (double v = std::ceil(lo - 1e-9); v <= std::floor(hi + 1e-9); v += 1.0) {
    (*point)[index] = v;
    enumerate(lp, integral, point, index + 1, any, best);
  }
}

double enumerated_optimum(const Model& model, bool* found) {
  const sankhya::StandardFormResult sf = sankhya::to_standard_form(model);
  CHECK(sf.ok);
  std::vector<bool> integral(sz(model.num_cols()), false);
  for (Int j = 0; j < model.num_cols(); ++j)
    integral[sz(j)] = model.col_type[sz(j)] != sankhya::VarType::kContinuous;

  std::vector<double> point(sz(sf.lp.num_cols()), 0.0);
  double best = kInf;
  *found = false;
  enumerate(sf.lp, integral, &point, 0, found, &best);
  // Back into the model's own sense, which is what solve_milp reports in.
  return sf.lp.objective_scale * best + sf.lp.objective_offset;
}

// Solve under one set of options and check the proved answer against the
// enumerated one.
void agrees_with_enumeration(const std::string& text, const char* label,
                             const BranchAndBoundOptions& options,
                             bool require_proof = true) {
  const Model model = read(text);
  bool feasible = false;
  const double truth = enumerated_optimum(model, &feasible);

  BranchAndBoundResult r = sankhya::solve_milp(model, options);

  if (!feasible) {
    // Nothing satisfies the rows, so the only correct answers are "infeasible"
    // or an honest failure to decide. An incumbent here would be one the model
    // does not contain.
    if (r.status == MilpStatus::kOptimal || r.status == MilpStatus::kFeasible) {
      std::fprintf(stderr, "%s: reported a solution for an infeasible model\n",
                   label);
      CHECK(false);
    }
    return;
  }

  if (r.status != MilpStatus::kOptimal) {
    // Failing to prove is allowed where the caller says so - a first-order node
    // solver stops on a tolerance and need not converge on every fixture. What
    // is never allowed is claiming a proof and being wrong, which is what the
    // rest of this function checks.
    if (!require_proof) return;
    std::fprintf(stderr, "%s: did not prove optimality (%s)\n", label,
                 sankhya::to_string(r.status).c_str());
    CHECK(false);
    return;
  }
  if (!sankhya_test::close(r.objective, truth, 1e-7)) {
    std::fprintf(stderr, "%s: proved %.12g, enumeration says %.12g\n", label,
                 r.objective, truth);
    CHECK(false);
    return;
  }
  // A proved optimum has to come with a point that is actually feasible and
  // actually has that value. The objective alone can be right while the vector
  // is not, and it is the vector that gets handed to whoever asked.
  const sankhya::StandardFormResult sf = sankhya::to_standard_form(model);
  CHECK(static_cast<Int>(r.x.size()) == sf.lp.num_cols());
  if (static_cast<Int>(r.x.size()) == sf.lp.num_cols()) {
    std::vector<double> scratch;
    double inf_norm = 0.0;
    sf.lp.primal_residual(r.x, &scratch, nullptr, &inf_norm);
    CHECK_NEAR(inf_norm, 0.0, 1e-6);
    for (Int j = 0; j < model.num_cols(); ++j) {
      if (model.col_type[sz(j)] == sankhya::VarType::kContinuous) continue;
      CHECK_NEAR(r.x[sz(j)] - std::round(r.x[sz(j)]), 0.0, 1e-6);
    }
    CHECK_NEAR(sf.lp.objective_scale * sf.lp.standard_objective(r.x) +
                   sf.lp.objective_offset,
               truth, 1e-7);
  }
}

BranchAndBoundOptions quiet_options() {
  BranchAndBoundOptions o;
  o.time_limit_seconds = 30.0;
  o.node_limit = 200000;
  return o;
}

// The same model under every combination of the switches that change what gets
// discarded. Each of these is a prune or a bound tightening, and a prune that is
// wrong is exactly the failure this file exists to catch - so they are varied
// against a fixed truth rather than trusted separately.
void agrees_under_every_pruning_option(const std::string& text,
                                       const char* label) {
  for (const bool integrality : {false, true}) {
    for (const bool reduced_cost : {false, true}) {
      for (const bool propagate : {false, true}) {
        for (const bool cuts : {false, true}) {
          BranchAndBoundOptions o = quiet_options();
          o.objective_integrality = integrality;
          o.reduced_cost_fixing = reduced_cost;
          o.node_propagation = propagate;
          o.root_cuts = cuts;
          const std::string named =
              std::string(label) + " [obj-int=" + (integrality ? "1" : "0") +
              " rcf=" + (reduced_cost ? "1" : "0") +
              " prop=" + (propagate ? "1" : "0") + " cuts=" + (cuts ? "1" : "0") +
              "]";
          agrees_with_enumeration(text, named.c_str(), o);
        }
      }
    }
  }
}

// The same models with the node relaxations solved by the first-order method
// rather than the simplex.
//
// The two differ in a way the tree has to respect. A simplex stops on an exact
// optimality test, so the objective at the vertex it stops on is the
// relaxation's optimum and is a lower bound on the subtree. A first-order
// method stops on a tolerance, and a point that is feasible but a little short
// of optimal has an objective ABOVE that optimum - so using it as a lower bound
// over-estimates the bound, and an over-estimated lower bound is how a prune
// discards the answer. The tree takes the dual objective instead, which is a
// bound by weak duality whether or not anything is optimal.
//
// Not required to prove optimality here: the point is that when it claims one,
// the claim is true.
void agrees_under_the_first_order_node_solver(const std::string& text,
                                              const char* label) {
  BranchAndBoundOptions o = quiet_options();
  o.node_solver = BranchAndBoundOptions::NodeSolver::kFirstOrder;
  // Gomory cuts are read off a tableau this solver does not have, and the
  // crossover seed is for the simplex.
  o.root_cuts = false;
  o.root_crossover = false;
  o.relaxation.tolerance = 1e-9;
  o.relaxation.max_iterations = 100000;
  agrees_with_enumeration(text, label, o, /*require_proof=*/false);
}

// A knapsack whose LP relaxation rounds the wrong way. This is the example from
// the guide: the relaxation says (2.857, 1.714), rounding gives 19, and the
// answer is (4, 0) at 20.
const char* kRoundsWrong =
    "NAME          ROUNDWRONG\n"
    "ROWS\n"
    " N  PROFIT\n"
    " L  CAP1\n"
    " L  CAP2\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X1        PROFIT          -5.0   CAP1             4.0\n"
    "    X1        CAP2             3.0\n"
    "    X2        PROFIT          -4.0   CAP1             5.0\n"
    "    X2        CAP2             2.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       CAP1            20.0   CAP2            12.0\n"
    "BOUNDS\n"
    " UI BND       X1               6.0\n"
    " UI BND       X2               6.0\n"
    "ENDATA\n";

// Every objective coefficient a whole number, which is what turns on the
// unit-strength cutoff. Small ranges so the tree has somewhere to go wrong.
const char* kIntegralObjective =
    "NAME          INTOBJ\n"
    "ROWS\n"
    " N  COST\n"
    " G  R1\n"
    " L  R2\n"
    " E  R3\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X1        COST             3.0   R1               2.0\n"
    "    X1        R2               1.0   R3               1.0\n"
    "    X2        COST            -7.0   R1               1.0\n"
    "    X2        R2               3.0\n"
    "    X3        COST             5.0   R1              -1.0\n"
    "    X3        R2               2.0   R3               1.0\n"
    "    X4        COST            -2.0   R2               1.0\n"
    "    X4        R3              -1.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       R1               3.0   R2              14.0\n"
    "    RHS       R3               2.0\n"
    "BOUNDS\n"
    " UI BND       X1               5.0\n"
    " UI BND       X2               5.0\n"
    " UI BND       X3               5.0\n"
    " UI BND       X4               5.0\n"
    "ENDATA\n";

// The same shape with fractional costs, so the unit cutoff must not fire. If it
// fires anyway this model is where it shows: the gap between two attainable
// objective values here is smaller than one.
const char* kFractionalObjective =
    "NAME          FRACOBJ\n"
    "ROWS\n"
    " N  COST\n"
    " G  R1\n"
    " L  R2\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X1        COST             1.5   R1               2.0\n"
    "    X1        R2               1.0\n"
    "    X2        COST            -2.25  R1               1.0\n"
    "    X2        R2               3.0\n"
    "    X3        COST             0.75  R1              -1.0\n"
    "    X3        R2               2.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       R1               3.0   R2              12.0\n"
    "BOUNDS\n"
    " UI BND       X1               4.0\n"
    " UI BND       X2               4.0\n"
    " UI BND       X3               4.0\n"
    "ENDATA\n";

// A maximisation, because the standard form negates c and the unit cutoff has
// to survive that. The negative of a whole number is a whole number, and this
// is the fixture that says so.
const char* kMaximise =
    "NAME          MAXOBJ\n"
    "OBJSENSE\n"
    "    MAX\n"
    "ROWS\n"
    " N  PROFIT\n"
    " L  CAP1\n"
    " L  CAP2\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X1        PROFIT           7.0   CAP1             3.0\n"
    "    X1        CAP2             1.0\n"
    "    X2        PROFIT           4.0   CAP1             1.0\n"
    "    X2        CAP2             2.0\n"
    "    X3        PROFIT           9.0   CAP1             4.0\n"
    "    X3        CAP2             3.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       CAP1            11.0   CAP2             9.0\n"
    "BOUNDS\n"
    " UI BND       X1               4.0\n"
    " UI BND       X2               4.0\n"
    " UI BND       X3               4.0\n"
    "ENDATA\n";

// Whole-number costs large enough that a unit is down at the noise floor of the
// arithmetic that produces a node bound. An LP objective carries something like
// 1e-9 of relative rounding, so at this size the slack the cutoff needs is
// itself about a unit - and the rule is written to turn itself off rather than
// claim a unit it cannot measure. This is the fixture that walks that branch.
//
// Same lesson as the two absolute tolerances elsewhere in this solver, each of
// which cost an answer: a tolerance on a quantity whose scale the model chooses
// has to scale with it.
const char* kLargeIntegralObjective =
    "NAME          BIGOBJ\n"
    "ROWS\n"
    " N  COST\n"
    " G  R1\n"
    " L  R2\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X1        COST    3000000001.0   R1               2.0\n"
    "    X1        R2               1.0\n"
    "    X2        COST   -7000000003.0   R1               1.0\n"
    "    X2        R2               3.0\n"
    "    X3        COST    5000000002.0   R1              -1.0\n"
    "    X3        R2               2.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       R1               3.0   R2              12.0\n"
    "BOUNDS\n"
    " UI BND       X1               4.0\n"
    " UI BND       X2               4.0\n"
    " UI BND       X3               4.0\n"
    "ENDATA\n";

// Rules that contradict each other. "Infeasible" is a claim, and a tree that
// makes it without a proof is the failure recorded in section 17 of the guide.
const char* kInfeasible =
    "NAME          NOANSWER\n"
    "ROWS\n"
    " N  COST\n"
    " L  R1\n"
    " G  R2\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    X1        COST             1.0   R1               1.0\n"
    "    X1        R2               1.0\n"
    "    X2        COST             1.0   R1               1.0\n"
    "    X2        R2               1.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS       R1               2.0   R2               5.0\n"
    "BOUNDS\n"
    " UI BND       X1               3.0\n"
    " UI BND       X2               3.0\n"
    "ENDATA\n";

// Random small integer programs. Four hand-written fixtures test what I thought
// to write down; these test what I did not. Fixed seed, because a test that
// fails only sometimes is a test nobody can act on.
void random_programs() {
  std::mt19937 rng(20260831u);
  std::uniform_int_distribution<int> coefficient(-4, 4);
  std::uniform_int_distribution<int> rhs(2, 14);
  std::uniform_int_distribution<int> bound(1, 3);

  for (int trial = 0; trial < 24; ++trial) {
    const int cols = 3 + (trial % 2);
    const int rows = 2 + (trial % 3);
    std::ostringstream mps;
    mps << "NAME          RAND" << trial << "\nROWS\n N  COST\n";
    for (int i = 0; i < rows; ++i)
      mps << ((trial + i) % 3 == 0 ? " G  R" : " L  R") << i << "\n";
    mps << "COLUMNS\n"
           "    MARKER                 'MARKER'                 'INTORG'\n";
    for (int j = 0; j < cols; ++j) {
      // Whole-number costs throughout, so the unit cutoff is live on every one
      // of these - that is the rule with the least margin for error.
      mps << "    X" << j << "        COST      " << coefficient(rng) << ".0\n";
      for (int i = 0; i < rows; ++i) {
        const int a = coefficient(rng);
        if (a == 0) continue;
        mps << "    X" << j << "        R" << i << "        " << a << ".0\n";
      }
    }
    mps << "    MARKER                 'MARKER'                 'INTEND'\nRHS\n";
    for (int i = 0; i < rows; ++i)
      mps << "    RHS       R" << i << "        " << rhs(rng) << ".0\n";
    mps << "BOUNDS\n";
    for (int j = 0; j < cols; ++j)
      mps << " UI BND       X" << j << "        " << bound(rng) << ".0\n";
    mps << "ENDATA\n";

    const std::string text = mps.str();
    const std::string label = "random " + std::to_string(trial);
    BranchAndBoundOptions on = quiet_options();
    BranchAndBoundOptions off = quiet_options();
    off.objective_integrality = false;
    agrees_with_enumeration(text, (label + " [obj-int=1]").c_str(), on);
    agrees_with_enumeration(text, (label + " [obj-int=0]").c_str(), off);
  }
}

// The other half of what the tree has to get right: what it says when it has
// not found anything. An objective of zero from a run with no incumbent scored
// as an exact answer on an instance whose optimum is zero, and no harness could
// tell the difference because the result struct could not either.
void reports_nothing_when_it_has_nothing() {
  const Model model = read(kMaximise);
  BranchAndBoundOptions o = quiet_options();
  // A budget nothing can finish in, so the run ends with the tree still open.
  o.time_limit_seconds = 0.0;
  o.node_limit = 0;
  o.root_cuts = false;
  o.rounding_heuristic = false;
  o.diving_heuristic = false;
  o.lp_diving = false;
  o.feasibility_pump = false;

  const BranchAndBoundResult r = sankhya::solve_milp(model, o);
  CHECK(r.status != MilpStatus::kOptimal);
  CHECK(r.status != MilpStatus::kFeasible);
  CHECK(r.x.empty());
  // Not zero. This model maximises, so the sense of "no answer" is minus
  // infinity; the CLI turns a non-finite objective into JSON null, which is
  // what a harness can test.
  CHECK(model.sense == ObjSense::kMaximize);
  CHECK(!std::isfinite(r.objective));
  CHECK(!std::isfinite(r.relative_gap));
}

// A dive, a pump and a rounding all report points they claim are feasible, and
// the tree takes them as incumbents. A heuristic that reports a point it has not
// checked poisons the bound that prunes everything else, so the answer has to
// survive every one of them being on.
void heuristics_do_not_poison_the_bound() {
  for (const char* text : {kRoundsWrong, kIntegralObjective, kMaximise}) {
    BranchAndBoundOptions o = quiet_options();
    o.rounding_heuristic = true;
    o.diving_heuristic = true;
    o.lp_diving = true;
    o.feasibility_pump = true;
    o.pump_retry_nodes = 1;
    o.lp_dive_retry_nodes = 1;
    agrees_with_enumeration(text, "every heuristic on", o);
  }
}

}  // namespace

int main() {
  agrees_under_every_pruning_option(kRoundsWrong, "rounds the wrong way");
  agrees_under_every_pruning_option(kIntegralObjective, "integral objective");
  agrees_under_every_pruning_option(kFractionalObjective, "fractional objective");
  agrees_under_every_pruning_option(kMaximise, "maximisation");
  agrees_under_every_pruning_option(kLargeIntegralObjective, "large integral objective");
  agrees_under_every_pruning_option(kInfeasible, "infeasible");
  agrees_under_the_first_order_node_solver(kRoundsWrong, "first-order nodes: rounds wrong");
  agrees_under_the_first_order_node_solver(kIntegralObjective, "first-order nodes: integral objective");
  agrees_under_the_first_order_node_solver(kFractionalObjective, "first-order nodes: fractional objective");
  agrees_under_the_first_order_node_solver(kMaximise, "first-order nodes: maximisation");
  agrees_under_the_first_order_node_solver(kInfeasible, "first-order nodes: infeasible");
  random_programs();
  reports_nothing_when_it_has_nothing();
  heuristics_do_not_poison_the_bound();
  return sankhya_test::finish("branch_and_bound");
}
