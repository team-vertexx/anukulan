#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/simplex.hpp"

namespace sankhya {

// Branch and bound for mixed-integer linear programs, solving each node's
// relaxation with the first-order method.
//
// Be clear about what this is. A serious MILP code solves node relaxations with
// a dual simplex warm-started from the parent basis, re-optimising in a handful
// of pivots. A first-order method has no basis to warm start from and gives a
// vertex only in the limit, so every node costs a full solve and integrality has
// to be read off a point that is merely close to a vertex. That is affordable on
// small problems and hopeless on large ones.
//
// It is here because the LP and MILP halves of the project need to be connected
// end to end before the simplex exists, and because a tree that is correct on
// small instances is what the simplex will later be dropped into. The honest
// scope is tens of integer variables, not thousands.
struct BranchAndBoundOptions {
  PdhgOptions relaxation;
  SimplexOptions simplex;

  // What solves a node's relaxation.
  //
  // The first-order method has no basis, so every node is a fresh solve and
  // integrality has to be read off a point that is only near a vertex. The
  // simplex has a basis, and branching changes one bound on one variable, which
  // cannot change a reduced cost - so the parent's basis is still dual feasible
  // and the child is a few dual pivots away. That is the difference between a
  // tree that is affordable and one that is not.
  //
  // Both are kept because the comparison is worth being able to run.
  enum class NodeSolver { kFirstOrder, kSimplex };
  NodeSolver node_solver = NodeSolver::kSimplex;

  // Iterations a single node's re-optimisation may take, as a multiple of the
  // row count. A warm started dual should need a handful of pivots; needing
  // thousands means it is stalling, and letting one node spend the solver's
  // whole iteration budget is how neos5 managed 16 nodes in 63 seconds. Past
  // the cap the node is reported as unsolved rather than pruned silently.
  Int node_iteration_factor = 0;  // 0 means no cap

  // The same, for the diving heuristic, which pins every integer column and so
  // presents the solver with a problem unlike the one it just did. It is a
  // heuristic and gets a short leash.
  Int dive_iteration_factor = 20;

  Int node_limit = 20000;
  double time_limit_seconds = 300.0;

  // A solution counts as integral when every integer variable is within this of
  // a whole number. It has to be looser than the relaxation tolerance, since the
  // relaxation only approaches a vertex.
  double integrality_tolerance = 1e-6;

  // Stop once the proven gap is this small, relative to the incumbent.
  double gap_tolerance = 1e-6;

  // Cutting planes, separated from the original rows at the root before the
  // tree starts. Gomory cuts would need a simplex tableau, which a first-order
  // method does not have; cover and MIR inequalities need only the rows and the
  // current point, so they are available now. See cuts.hpp.
  bool root_cuts = true;
  Int cut_rounds = 8;
  // Rounds in which Gomory cuts are separated. They are read off the tableau,
  // and after the first round that tableau describes a problem that is mostly
  // previous cuts - so each round derives cuts from cuts, and the rounding
  // arguments they rest on stop being exact.
  //
  // gt2 is what that looks like. With Gomory on in every round its root bound
  // goes 13,460, 20,539, 20,857, 21,094, then 26,833 - past the optimum of
  // 21,166 - and then back down to 21,110. A bound that falls after a cut is
  // added is not a bound; the relaxation had been cut into something else.
  // Gomory cuts are off by default, and that is a measurement rather than a
  // doubt about them. They tighten the root bound on every one of the seven
  // MIPLIB instances here - khb05250 from 95.9M to 105.7M against an optimum of
  // 106.9M, p0201 from 7,125 to 7,199, gt2 from 20,593 to 20,862 - and then
  // lose the tree. After 25 seconds gt2 goes from 5.556% to 11.112%, mas76 from
  // 0.562% to 1.593%, gen-ip054 from 0.528% to 0.847%, and the other four are
  // unchanged at zero.
  //
  // cuts.hpp already said why: a cut row costs something in every node below
  // it, on every iteration, and a better root bound has to pay for that out of
  // nodes not explored. A tighter relaxation that is explored less is not
  // obviously a better solver, and here it measurably is not.
  //
  // What would change this is cut management the tree does not have yet -
  // ageing cuts out when they stop binding, and keeping them in a pool rather
  // than in the matrix. Worth doing; not done.
  //
  // One thing measured on the way and worth not repeating: separating cover and
  // MIR cuts from `working`, which by the later rounds is mostly cuts, was
  // briefly blamed for a root bound that ran past the optimum and restricted to
  // the model's own rows. It was not the cause - Gomory was - and the
  // restriction cost gt2 its exact answer, 0.000% to 5.556%, and cut the number
  // of cuts found from 77 to 32. A MIR of a valid inequality is valid; two
  // things were changed at once and the wrong one was blamed.
  // Reduced-cost fixing. Once there is an incumbent, a node's own LP bound and
  // its reduced costs together say how far a nonbasic variable can move before
  // the subtree stops being worth exploring:
  //
  //   at its lower bound, d_j > 0:  x_j <= l_j + (incumbent - z_node) / d_j
  //   at its upper bound, d_j < 0:  x_j >= u_j - (incumbent - z_node) / |d_j|
  //
  // Moving further than that costs more than the incumbent already achieves, so
  // nothing below this node needs the rest of the range. The bound is tightened
  // for the children only - it is valid in this subtree, not globally, because
  // it is derived from this node's bound.
  //
  // Costs one transpose product per node on top of the relaxation, which is
  // small against a node solve, and it gets stronger as the tree deepens and
  // the gap closes.
  bool reduced_cost_fixing = true;

  bool gomory_cuts = false;

  // Rounds in which Gomory cuts are separated when they are on at all. They are
  // read off the tableau, and after the first round that tableau describes a
  // problem that is mostly previous cuts, so each round derives cuts from cuts
  // and the rounding arguments stop being exact. On gt2 the root bound reaches
  // 20,592 with none, 20,592 with one round and 20,861 with two, all rising
  // monotonically; from three rounds on it passes the optimum of 21,166 - 26,833
  // - and then falls back to 21,110. A bound that falls after a cut is added is
  // not a bound.
  Int gomory_rounds = 2;
  Int cuts_per_round = 60;

  // Branching rule. Most-fractional is the classic bad default, kept so its
  // cost can be measured rather than asserted. Pseudocost keeps a running
  // average of how much the objective actually degraded the last time each
  // variable was branched on, which is a far better predictor and costs nothing
  // beyond bookkeeping.
  enum class Branching { kMostFractional, kPseudocost };
  Branching branching = Branching::kPseudocost;

  // Try rounding the relaxation to an integer point, and then dive by fixing one
  // variable at a time. Without a heuristic there is no incumbent until the
  // depth-first search happens to reach a feasible leaf, and on some instances
  // that never happens inside the time limit.
  bool rounding_heuristic = true;
  bool diving_heuristic = true;
  Int diving_max_depth = 40;

  // Start each child from its parent's solution. The first-order method has no
  // basis to inherit, but the parent's point is still a much better starting
  // guess than the origin.
  bool warm_start = true;

  bool verbose = false;
};

enum class MilpStatus {
  kOptimal,
  kFeasible,     // an incumbent was found, optimality not proven
  kInfeasible,
  kNodeLimit,
  kTimeLimit,
  kRelaxationFailed,
};

std::string to_string(MilpStatus status);

struct BranchAndBoundResult {
  MilpStatus status = MilpStatus::kRelaxationFailed;
  std::vector<double> x;
  double objective = 0.0;   // in the model's own sense
  double dual_bound = 0.0;  // best bound over the open tree
  double relative_gap = 0.0;

  Int nodes = 0;
  Int max_depth = 0;
  Int relaxations_solved = 0;
  Int incumbents_found = 0;
  Int nodes_proved_infeasible = 0;
  Int nodes_relaxation_failed = 0;
  Int heuristic_successes = 0;
  // Bounds tightened by reduced-cost fixing, and how many closed a range to nothing.
  Int reduced_cost_tightenings = 0;
  Int reduced_cost_fixings = 0;
  Int dives_run = 0;
  Int cuts_added = 0;
  // Of those, how many came off the tableau rather than the model's rows.
  Int gomory_cuts_added = 0;
  // Set when a round of cuts lowered the root bound, which valid cuts cannot
  // do. The round is rolled back and cutting stops.
  bool cuts_rolled_back = false;
  // Nodes whose relaxation started from the parent's basis rather than from
  // nothing. Close to the node count is what this is supposed to look like.
  Int warm_started_nodes = 0;
  Int simplex_iterations = 0;
  double root_bound_before_cuts = 0.0;
  double root_bound_after_cuts = 0.0;
  double solve_seconds = 0.0;
  std::string message;
};

BranchAndBoundResult solve_milp(const Model& model,
                                const BranchAndBoundOptions& options = {});

}  // namespace sankhya
