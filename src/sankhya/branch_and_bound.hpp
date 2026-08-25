#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/pdhg.hpp"

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

  Int node_limit = 20000;
  double time_limit_seconds = 300.0;

  // A solution counts as integral when every integer variable is within this of
  // a whole number. It has to be looser than the relaxation tolerance, since the
  // relaxation only approaches a vertex.
  double integrality_tolerance = 1e-6;

  // Stop once the proven gap is this small, relative to the incumbent.
  double gap_tolerance = 1e-6;

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
  double solve_seconds = 0.0;
  std::string message;
};

BranchAndBoundResult solve_milp(const Model& model,
                                const BranchAndBoundOptions& options = {});

}  // namespace sankhya
