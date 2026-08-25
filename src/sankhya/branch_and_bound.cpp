#include "sankhya/branch_and_bound.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "sankhya/standard_form.hpp"

namespace sankhya {
namespace {

struct Node {
  std::vector<double> lower;
  std::vector<double> upper;
  double parent_bound = -kInf;
  Int depth = 0;
  // The parent's relaxation solution, used as a starting guess.
  std::vector<double> warm_x;
};

double fractionality(double v) { return std::fabs(v - std::round(v)); }

// Provable infeasibility from bounds alone, by interval arithmetic on each row.
//
// This matters more here than it would in a simplex-based tree. A dual simplex
// reports an infeasible node immediately; a first-order method has no way to
// tell infeasibility from slow convergence, so it runs to its iteration limit
// and then the node has to be pruned on a guess. Guessing is unsafe - a node
// pruned because its relaxation merely stalled could have contained the optimum,
// which makes the final bound worthless as a proof.
//
// Row activity bounds are exact and cheap, so whatever this catches is pruned
// with a proof rather than a shrug.
bool provably_infeasible(const StandardLp& lp, const std::vector<double>& lower,
                         const std::vector<double>& upper) {
  for (std::size_t j = 0; j < lower.size(); ++j) {
    if (lower[j] > upper[j]) return true;
  }
  for (Int i = 0; i < lp.k.rows(); ++i) {
    double min_activity = 0.0;
    double max_activity = 0.0;
    bool min_finite = true;
    bool max_finite = true;
    for (Int e = lp.k.row_begin(i); e < lp.k.row_end(i); ++e) {
      const double a = lp.k.value()[sz(e)];
      const std::size_t j = sz(lp.k.index()[sz(e)]);
      const double lo = (a > 0.0) ? lower[j] : upper[j];
      const double hi = (a > 0.0) ? upper[j] : lower[j];
      if (std::isinf(lo)) {
        min_finite = false;
      } else if (min_finite) {
        min_activity += a * lo;
      }
      if (std::isinf(hi)) {
        max_finite = false;
      } else if (max_finite) {
        max_activity += a * hi;
      }
      if (!min_finite && !max_finite) break;
    }
    const double q = lp.q[sz(i)];
    // Every standard-form row is either an equality or a >= row.
    if (max_finite && max_activity < q - 1e-9) return true;
    if (i < lp.num_equalities && min_finite && min_activity > q + 1e-9) return true;
  }
  return false;
}

}  // namespace

std::string to_string(MilpStatus status) {
  switch (status) {
    case MilpStatus::kOptimal:
      return "optimal";
    case MilpStatus::kFeasible:
      return "feasible";
    case MilpStatus::kInfeasible:
      return "infeasible";
    case MilpStatus::kNodeLimit:
      return "node limit";
    case MilpStatus::kTimeLimit:
      return "time limit";
    case MilpStatus::kRelaxationFailed:
      return "relaxation failed";
  }
  return "unknown";
}

BranchAndBoundResult solve_milp(const Model& model,
                                const BranchAndBoundOptions& options) {
  const auto start_time = std::chrono::steady_clock::now();
  auto elapsed = [&start_time]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
        .count();
  };

  BranchAndBoundResult result;

  const StandardFormResult sf = to_standard_form(model);
  if (!sf.ok) {
    result.message = sf.error;
    return result;
  }

  std::vector<Int> integer_columns;
  for (Int j = 0; j < model.num_cols(); ++j) {
    if (model.col_type[sz(j)] == VarType::kInteger) integer_columns.push_back(j);
  }

  // A minimisation in standard form; the model's own sense is restored at the
  // end through objective_scale. Bounds compare in the standard direction.
  const bool maximizing = model.sense == ObjSense::kMaximize;
  auto to_model_objective = [&](double standard) {
    return sf.lp.objective_scale * standard + sf.lp.objective_offset;
  };

  double incumbent = kInf;  // best standard-form objective found
  std::vector<double> incumbent_x;

  std::vector<Node> stack;
  Node root;
  root.lower = sf.lp.lower;
  root.upper = sf.lp.upper;
  stack.push_back(std::move(root));

  auto solve_node = [&](const Node& node, PdhgResult* out) {
    StandardLp lp = sf.lp;
    lp.lower = node.lower;
    lp.upper = node.upper;
    *out = solve_pdhg(lp, options.relaxation);
    result.relaxations_solved++;
    return out->status == PdhgStatus::kOptimal;
  };

  MilpStatus status = MilpStatus::kInfeasible;
  double open_bound = -kInf;

  while (!stack.empty()) {
    if (result.nodes >= options.node_limit) {
      status = MilpStatus::kNodeLimit;
      break;
    }
    if (elapsed() > options.time_limit_seconds) {
      status = MilpStatus::kTimeLimit;
      break;
    }

    Node node = std::move(stack.back());
    stack.pop_back();
    result.nodes++;
    result.max_depth = std::max(result.max_depth, node.depth);

    // Nothing below this node can beat the incumbent.
    if (node.parent_bound >= incumbent) continue;

    if (provably_infeasible(sf.lp, node.lower, node.upper)) {
      result.nodes_proved_infeasible++;
      continue;
    }

    PdhgResult relaxation;
    if (!solve_node(node, &relaxation)) {
      // Treat a node whose relaxation did not solve as pruned, and say so, since
      // silently dropping subtrees would make the reported bound a lie.
      result.nodes_relaxation_failed++;
      continue;
    }

    const double node_objective = sf.lp.standard_objective(relaxation.x);
    if (node_objective >= incumbent) continue;  // bound prune
    if (stack.empty()) open_bound = node_objective;

    // Most fractional integer variable.
    Int branch_column = -1;
    double worst = options.integrality_tolerance;
    for (const Int j : integer_columns) {
      const double f = fractionality(relaxation.x[sz(j)]);
      if (f > worst) {
        worst = f;
        branch_column = j;
      }
    }

    if (branch_column < 0) {
      // Integral: a new incumbent.
      incumbent = node_objective;
      incumbent_x = relaxation.x;
      result.incumbents_found++;
      if (options.verbose) {
        std::printf("  node %6d depth %3d  incumbent %.10e\n", result.nodes,
                    node.depth, to_model_objective(incumbent));
      }
      continue;
    }

    const double value = relaxation.x[sz(branch_column)];
    const double floor_value = std::floor(value);
    const double ceil_value = std::ceil(value);

    // Depth first, and the child nearer the relaxation value is explored first,
    // because it is the likelier place to find a feasible point early.
    const bool up_first = (value - floor_value) > 0.5;
    for (int which = 0; which < 2; ++which) {
      const bool take_up = (which == 0) ? !up_first : up_first;
      Node child;
      child.lower = node.lower;
      child.upper = node.upper;
      child.depth = node.depth + 1;
      child.parent_bound = node_objective;
      if (options.warm_start) child.warm_x = relaxation.x;

      if (take_up) {
        child.lower[sz(branch_column)] =
            std::fmax(child.lower[sz(branch_column)], ceil_value);
      } else {
        child.upper[sz(branch_column)] =
            std::fmin(child.upper[sz(branch_column)], floor_value);
      }
      if (child.lower[sz(branch_column)] > child.upper[sz(branch_column)]) continue;
      stack.push_back(std::move(child));
    }
  }

  if (!incumbent_x.empty()) {
    result.x = incumbent_x;
    result.objective = to_model_objective(incumbent);
    if (stack.empty() && status == MilpStatus::kInfeasible) {
      status = MilpStatus::kOptimal;
      result.dual_bound = result.objective;
      result.relative_gap = 0.0;
    } else {
      if (status == MilpStatus::kInfeasible) status = MilpStatus::kFeasible;
      double best_open = incumbent;
      for (const Node& n : stack) best_open = std::fmin(best_open, n.parent_bound);
      result.dual_bound = to_model_objective(best_open);
      result.relative_gap = std::fabs(result.objective - result.dual_bound) /
                            std::fmax(1e-10, std::fabs(result.objective));
    }
  } else if (status == MilpStatus::kInfeasible) {
    result.objective = maximizing ? -kInf : kInf;
  }

  (void)open_bound;
  if (result.nodes_relaxation_failed > 0) {
    result.message =
        std::to_string(result.nodes_relaxation_failed) +
        " node relaxation(s) hit their limit without converging and were pruned "
        "without proof, so optimality here is not certified";
    if (status == MilpStatus::kOptimal) status = MilpStatus::kFeasible;
  }
  result.status = status;
  result.solve_seconds = elapsed();
  return result;
}

}  // namespace sankhya
