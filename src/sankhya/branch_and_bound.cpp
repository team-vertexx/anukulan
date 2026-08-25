#include "sankhya/branch_and_bound.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "sankhya/cuts.hpp"
#include "sankhya/standard_form.hpp"

namespace sankhya {
namespace {

struct Node {
  std::vector<double> lower;
  std::vector<double> upper;
  double parent_bound = -kInf;
  Int depth = 0;
  // Which branch produced this node, so its pseudocost can be updated once its
  // own relaxation value is known.
  Int branch_column = -1;
  bool branch_up = false;
  double branch_fraction = 0.0;
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
// Tighten variable bounds by interval arithmetic on each row, repeated to a
// fixpoint. Returns false when infeasibility is proved.
//
// This is the propagation half of the fix-and-propagate heuristic of Kempke and
// Koch (arXiv:2503.10344), which is written for exactly our situation: deriving
// integer solutions from a first-order LP solution that is only roughly
// accurate. Fixing one variable at a time and propagating between fixes catches
// a doomed rounding immediately, and often fixes further variables by inference
// at no cost, where rounding everything at once and checking afterwards learns
// nothing until the end.
bool propagate_bounds(const StandardLp& lp, const std::vector<bool>& integral,
                      std::vector<double>* lower, std::vector<double>* upper,
                      int max_rounds) {
  for (int round = 0; round < max_rounds; ++round) {
    bool changed = false;
    for (Int i = 0; i < lp.k.rows(); ++i) {
      // Activity bounds for the row, and whether either end is finite.
      double min_activity = 0.0;
      double max_activity = 0.0;
      Int min_infinite = 0;
      Int max_infinite = 0;
      for (Int e = lp.k.row_begin(i); e < lp.k.row_end(i); ++e) {
        const double a = lp.k.value()[sz(e)];
        const std::size_t j = sz(lp.k.index()[sz(e)]);
        const double lo = (a > 0.0) ? (*lower)[j] : (*upper)[j];
        const double hi = (a > 0.0) ? (*upper)[j] : (*lower)[j];
        if (std::isinf(lo)) ++min_infinite; else min_activity += a * lo;
        if (std::isinf(hi)) ++max_infinite; else max_activity += a * hi;
      }
      const double q = lp.q[sz(i)];
      if (max_infinite == 0 && max_activity < q - 1e-9) return false;
      if (i < lp.num_equalities && min_infinite == 0 && min_activity > q + 1e-9)
        return false;

      for (Int e = lp.k.row_begin(i); e < lp.k.row_end(i); ++e) {
        const double a = lp.k.value()[sz(e)];
        if (std::fabs(a) < 1e-12) continue;
        const std::size_t j = sz(lp.k.index()[sz(e)]);
        const double lo = (a > 0.0) ? (*lower)[j] : (*upper)[j];
        const double hi = (a > 0.0) ? (*upper)[j] : (*lower)[j];

        // a_ij x_j >= q - (largest the rest of the row can be)
        if (max_infinite == 0 || (max_infinite == 1 && std::isinf(hi))) {
          const double rest = max_infinite == 0 ? max_activity - a * hi : max_activity;
          const double limit = (q - rest) / a;
          if (a > 0.0) {
            double v = limit;
            if (integral[j]) v = std::ceil(v - 1e-9);
            if (v > (*lower)[j] + 1e-9) { (*lower)[j] = v; changed = true; }
          } else {
            double v = limit;
            if (integral[j]) v = std::floor(v + 1e-9);
            if (v < (*upper)[j] - 1e-9) { (*upper)[j] = v; changed = true; }
          }
        }
        // Equalities bind from the other side too.
        if (i < lp.num_equalities &&
            (min_infinite == 0 || (min_infinite == 1 && std::isinf(lo)))) {
          const double rest = min_infinite == 0 ? min_activity - a * lo : min_activity;
          const double limit = (q - rest) / a;
          if (a > 0.0) {
            double v = limit;
            if (integral[j]) v = std::floor(v + 1e-9);
            if (v < (*upper)[j] - 1e-9) { (*upper)[j] = v; changed = true; }
          } else {
            double v = limit;
            if (integral[j]) v = std::ceil(v - 1e-9);
            if (v > (*lower)[j] + 1e-9) { (*lower)[j] = v; changed = true; }
          }
        }
        if ((*lower)[j] > (*upper)[j] + 1e-7) return false;
      }
    }
    if (!changed) break;
  }
  return true;
}

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
  std::vector<bool> is_integral(sz(model.num_cols()), false);
  for (Int j = 0; j < model.num_cols(); ++j) {
    if (model.col_type[sz(j)] == VarType::kInteger) {
      integer_columns.push_back(j);
      is_integral[sz(j)] = true;
    }
  }

  // A minimisation in standard form; the model's own sense is restored at the
  // end through objective_scale. Bounds compare in the standard direction.
  const bool maximizing = model.sense == ObjSense::kMaximize;
  auto to_model_objective = [&](double standard) {
    return sf.lp.objective_scale * standard + sf.lp.objective_offset;
  };

  double incumbent = kInf;  // best standard-form objective found
  std::vector<double> incumbent_x;

  // Pseudocosts: for each integer column, the running average objective
  // degradation per unit of fractionality, measured separately for rounding the
  // variable down and rounding it up. Counts track how many observations back
  // each average, so an untried variable can be told from a cheap one.
  std::vector<double> pseudo_down(sz(model.num_cols()), 0.0);
  std::vector<double> pseudo_up(sz(model.num_cols()), 0.0);
  std::vector<Int> pseudo_down_n(sz(model.num_cols()), 0);
  std::vector<Int> pseudo_up_n(sz(model.num_cols()), 0);

  // Root cut loop: solve, separate, add, repeat. Cuts are kept for the whole
  // tree, so a round that tightens the root tightens every node below it.
  StandardLp working = sf.lp;
  if (options.root_cuts && !integer_columns.empty()) {
    PdhgOptions root_options = options.relaxation;
    for (Int round = 0; round < options.cut_rounds; ++round) {
      const PdhgResult root_solve = solve_pdhg(working, root_options);
      result.relaxations_solved++;
      if (root_solve.status != PdhgStatus::kOptimal) break;
      if (round == 0) {
        result.root_bound_before_cuts =
            sf.lp.objective_scale * sf.lp.standard_objective(root_solve.x) +
            sf.lp.objective_offset;
      }
      result.root_bound_after_cuts =
          sf.lp.objective_scale * sf.lp.standard_objective(root_solve.x) +
          sf.lp.objective_offset;

      CutOptions cut_options;
      cut_options.max_cuts = options.cuts_per_round;
      const std::vector<Cut> cuts =
          separate_cuts(working, is_integral, root_solve.x, cut_options);
      if (cuts.empty()) break;
      working = append_cuts(working, cuts);
      result.cuts_added += static_cast<Int>(cuts.size());
    }
  }

  std::vector<Node> stack;
  Node root;
  root.lower = working.lower;
  root.upper = working.upper;
  stack.push_back(std::move(root));

  auto try_incumbent = [&](const std::vector<double>& candidate) {
    // Accepts a point only if it is genuinely feasible and genuinely better.
    // A heuristic that reports a point it has not checked is worse than no
    // heuristic, because it poisons the bound that prunes the whole tree.
    for (const Int j : integer_columns) {
      if (fractionality(candidate[sz(j)]) > options.integrality_tolerance) return false;
    }
    for (std::size_t j = 0; j < candidate.size(); ++j) {
      if (candidate[j] < working.lower[j] - 1e-7) return false;
      if (candidate[j] > working.upper[j] + 1e-7) return false;
    }
    std::vector<double> scratch;
    double inf_norm = 0.0;
    working.primal_residual(candidate, &scratch, nullptr, &inf_norm);
    if (inf_norm > 1e-6) return false;

    const double value = working.standard_objective(candidate);
    if (value >= incumbent) return false;
    incumbent = value;
    incumbent_x = candidate;
    result.incumbents_found++;
    return true;
  };

  auto solve_node = [&](const Node& node, PdhgResult* out) {
    StandardLp lp = working;
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

    if (provably_infeasible(working, node.lower, node.upper)) {
      result.nodes_proved_infeasible++;
      continue;
    }

    PdhgResult relaxation;
    if (solve_node(node, &relaxation) == false &&
        relaxation.status == PdhgStatus::kPrimalInfeasible) {
      // Proved empty, so nothing below this node exists. Safe to prune.
      result.nodes_proved_infeasible++;
      continue;
    }
    if (relaxation.status != PdhgStatus::kOptimal) {
      // Treat a node whose relaxation did not solve as pruned, and say so, since
      // silently dropping subtrees would make the reported bound a lie.
      result.nodes_relaxation_failed++;
      continue;
    }

    const double node_objective = working.standard_objective(relaxation.x);

    // Update the pseudocost for the branch that created this node: how much did
    // the objective actually degrade, per unit of fractionality given up.
    if (node.branch_column >= 0 && node.branch_fraction > 1e-9 &&
        node.parent_bound > -kInf) {
      const double degradation =
          std::fmax(0.0, node_objective - node.parent_bound) / node.branch_fraction;
      const std::size_t c = sz(node.branch_column);
      if (node.branch_up) {
        pseudo_up[c] = (pseudo_up[c] * static_cast<double>(pseudo_up_n[c]) +
                        degradation) /
                       static_cast<double>(pseudo_up_n[c] + 1);
        pseudo_up_n[c]++;
      } else {
        pseudo_down[c] = (pseudo_down[c] * static_cast<double>(pseudo_down_n[c]) +
                          degradation) /
                         static_cast<double>(pseudo_down_n[c] + 1);
        pseudo_down_n[c]++;
      }
    }

    if (node_objective >= incumbent) continue;  // bound prune
    if (stack.empty()) open_bound = node_objective;

    // Before branching, try to turn this relaxation into a feasible point.
    //
    // Cheap first: round everything at once and check. It costs nothing and
    // succeeds surprisingly often on problems whose relaxation is already nearly
    // integral.
    if (options.rounding_heuristic) {
      std::vector<double> rounded = relaxation.x;
      for (const Int j : integer_columns) rounded[sz(j)] = std::round(rounded[sz(j)]);
      for (std::size_t j = 0; j < rounded.size(); ++j) {
        rounded[j] = std::fmin(std::fmax(rounded[j], node.lower[j]), node.upper[j]);
      }
      if (try_incumbent(rounded)) result.heuristic_successes++;
    }

    // Then fix-and-propagate, which costs one extra relaxation solve, so it runs
    // at the root and then only periodically.
    const bool run_dive = options.diving_heuristic && incumbent_x.empty() &&
                          (node.depth == 0 || result.nodes % 25 == 0);
    if (run_dive) {
      result.dives_run++;
      std::vector<double> lo = node.lower;
      std::vector<double> hi = node.upper;

      // Furthest from integrality first, following Kempke and Koch: those are
      // the most constrained choices, so a bad one is discovered early.
      std::vector<Int> order(integer_columns);
      std::sort(order.begin(), order.end(), [&](Int a, Int b) {
        return fractionality(relaxation.x[sz(a)]) > fractionality(relaxation.x[sz(b)]);
      });

      bool ok = propagate_bounds(working, is_integral, &lo, &hi, 4);
      for (std::size_t idx = 0; ok && idx < order.size(); ++idx) {
        const std::size_t j = sz(order[idx]);
        if (hi[j] - lo[j] < 0.5) continue;  // propagation already fixed it
        const double target = std::fmin(std::fmax(std::round(relaxation.x[j]), lo[j]),
                                        hi[j]);
        const double other = (target > relaxation.x[j]) ? std::fmax(lo[j], target - 1.0)
                                                        : std::fmin(hi[j], target + 1.0);
        bool fixed = false;
        for (const double value : {target, other}) {
          std::vector<double> trial_lo = lo;
          std::vector<double> trial_hi = hi;
          trial_lo[j] = value;
          trial_hi[j] = value;
          if (propagate_bounds(working, is_integral, &trial_lo, &trial_hi, 4)) {
            lo.swap(trial_lo);
            hi.swap(trial_hi);
            fixed = true;
            break;
          }
        }
        if (!fixed) ok = false;  // both roundings are dead, give up on this dive
      }

      if (ok) {
        // The integers are pinned; one solve settles the continuous variables.
        StandardLp dive = working;
        dive.lower = lo;
        dive.upper = hi;
        PdhgOptions dive_options = options.relaxation;
        dive_options.time_limit_seconds =
            std::fmin(dive_options.time_limit_seconds, 5.0);
        const PdhgResult dive_result = solve_pdhg(dive, dive_options);
        result.relaxations_solved++;
        if (dive_result.status == PdhgStatus::kOptimal) {
          std::vector<double> candidate = dive_result.x;
          for (const Int j : integer_columns)
            candidate[sz(j)] = std::round(candidate[sz(j)]);
          if (try_incumbent(candidate)) result.heuristic_successes++;
        }
      }
    }

    // Pick the variable to branch on.
    Int branch_column = -1;
    if (options.branching == BranchAndBoundOptions::Branching::kPseudocost) {
      double best = -1.0;
      for (const Int j : integer_columns) {
        const double v = relaxation.x[sz(j)];
        const double f = fractionality(v);
        if (f <= options.integrality_tolerance) continue;
        const double down_frac = v - std::floor(v);
        const double up_frac = std::ceil(v) - v;
        // An unmeasured direction gets an optimistic unit cost, so every
        // variable is tried once before the averages decide anything.
        const double dc = pseudo_down_n[sz(j)] ? pseudo_down[sz(j)] : 1.0;
        const double uc = pseudo_up_n[sz(j)] ? pseudo_up[sz(j)] : 1.0;
        // The usual product score, with a small linear term so that a variable
        // whose one side is free still gets ranked.
        const double down = dc * down_frac;
        const double up = uc * up_frac;
        const double score = std::fmax(down, 1e-6) * std::fmax(up, 1e-6) +
                             1e-6 * (down + up);
        if (score > best) {
          best = score;
          branch_column = j;
        }
      }
    } else {
      double worst = options.integrality_tolerance;
      for (const Int j : integer_columns) {
        const double f = fractionality(relaxation.x[sz(j)]);
        if (f > worst) {
          worst = f;
          branch_column = j;
        }
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
      child.branch_column = branch_column;
      child.branch_up = take_up;
      child.branch_fraction =
          take_up ? (ceil_value - value) : (value - floor_value);
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
    // "Infeasible" is a claim, and it is only true when every node was pruned
    // with a proof. If any relaxation ran out of budget without converging,
    // what we actually know is that we failed to solve the problem, and saying
    // infeasible instead would be reporting a wrong answer confidently. The
    // refinery MILP hit exactly this: its root relaxation does not converge, and
    // an earlier version of this code called the instance infeasible while
    // HiGHS solved it to optimality.
    if (result.nodes_relaxation_failed > 0) {
      status = MilpStatus::kRelaxationFailed;
      result.message =
          std::to_string(result.nodes_relaxation_failed) +
          " relaxation(s) did not converge, so infeasibility was never proved";
    }
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
