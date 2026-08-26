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
  Int dives_run = 0;
  Int cuts_added = 0;
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
