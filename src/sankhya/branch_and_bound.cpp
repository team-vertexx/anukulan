#include "sankhya/branch_and_bound.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "sankhya/crossover.hpp"
#include "sankhya/cuts.hpp"
#include "sankhya/simplex.hpp"
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
  // Forrest's estimate of the best integer objective obtainable below this
  // node: the parent's relaxation value plus, for every column still
  // fractional there, the cheaper of the two roundings priced by its
  // pseudocost. See the node selection option in branch_and_bound.hpp.
  double estimate = -kInf;
  // The parent's relaxation solution, used as a starting guess. Only the
  // first-order node solver consumes it, so only that solver pays to carry it -
  // it is n doubles on every open node, which is the largest single thing a
  // node holds and is pure waste under the simplex, where what a child inherits
  // is the basis below.
  std::vector<double> warm_x;
  // And the parent's basis, which is what actually makes a child cheap.
  std::vector<Int> basic;
  std::vector<VarStatus> basis_status;
};

// What a node's relaxation produced, whichever solver produced it.
struct NodeSolution {
  bool optimal = false;
  bool proved_infeasible = false;
  bool warm = false;
  // The lower bound this node establishes on its own subtree, which is not the
  // same thing as the objective at the point the solver returned. See where it
  // is set in solve_node: for the simplex those two coincide, and for a
  // first-order method they do not.
  double bound = -kInf;
  std::vector<double> x;
  std::vector<double> y;  // row duals, for reduced-cost fixing
  std::vector<Int> basic;
  std::vector<VarStatus> basis_status;
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
  // A crossed bound is only a proof of infeasibility if it is a real crossing.
  // Propagation writes bounds computed as (q - rest) / a and accepts a box
  // whose bounds cross by up to 1e-7; this used to reject one that crossed by
  // anything at all, including by the last bit of a double.
  //
  // On fiber that difference lost the answer. A column came out of propagation
  // with lower 6.0000000000000018 against upper 6 - a gap of 1.8e-15, which is
  // rounding and nothing else - and the subtree holding the optimum was
  // discarded as proved infeasible. The solver then reported 652,748.78 as a
  // proved optimum against a true 405,935.18, with a matching dual bound and no
  // warning, because every step after the bad prune was consistent with it.
  //
  // The two tests have to agree about what an empty box is. This one now uses
  // the same tolerance propagation does, scaled by the bound's own magnitude.
  for (std::size_t j = 0; j < lower.size(); ++j) {
    const double give = 1e-7 * std::fmax(1.0, std::fmax(std::fabs(lower[j]),
                                                        std::fabs(upper[j])));
    if (lower[j] > upper[j] + give) return true;
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
    // The tolerance has to scale with the row, and this is the whole of the
    // reason: an absolute 1e-9 is below the noise floor of the arithmetic that
    // produced these numbers. A row whose activity is around 1e6 carries about
    // 1e-10 of rounding per term in double precision, and a few hundred terms
    // put that past 1e-9 - so a row that is exactly tight computes as violated
    // and the whole subtree under it is discarded as proved infeasible.
    //
    // fiber is where this surfaced. With cuts appended - which raise the
    // coefficient magnitudes - a node holding the optimum was declared
    // infeasible here, and the solver reported 652,748.78 as a proved optimum
    // against a true 405,935.18. Nothing downstream can catch that: the answer
    // is confident, self-consistent and wrong.
    //
    // Same shape as the presolve bug earlier in this repository, where a 1e-9
    // bound relaxation manufactured forcing rows. An absolute tolerance on a
    // quantity whose scale the caller chooses is always this bug waiting.
    const double scale =
        std::fmax(1.0, std::fmax(std::fabs(q), max_finite ? std::fabs(max_activity)
                                                          : 0.0));
    if (max_finite && max_activity < q - 1e-9 * scale) return true;
    const double min_scale =
        std::fmax(1.0, std::fmax(std::fabs(q), min_finite ? std::fabs(min_activity)
                                                          : 0.0));
    if (i < lp.num_equalities && min_finite && min_activity > q + 1e-9 * min_scale)
      return true;
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

  // Whether every feasible objective value is a whole number.
  //
  // It is, when the objective touches no continuous column and every
  // coefficient it does touch is an integer - then c'x is a sum of integers for
  // any integer-feasible x. Both halves have to hold: one continuous column
  // with a cost makes the objective continuous however integral the rest is.
  //
  // What this buys is a stronger cutoff. Without it a node is pruned when its
  // bound reaches the incumbent, which throws away nothing but also catches
  // nothing until the bound has climbed the whole way. With it, a node whose
  // bound is anywhere above incumbent - 1 is dead: the best whole number in
  // that subtree is already the incumbent's own value. The prune arrives up to
  // a full unit earlier and it is exact, not a tolerance - there is no version
  // of this that discards a better answer, which is the property that matters
  // here more than the strength.
  //
  // Negation does not disturb it, so a maximisation is covered: standard-form c
  // is -c for those, and the negative of a whole number is a whole number. The
  // offset does not disturb it either, because everything below compares two
  // objective values and the offset cancels.
  //
  // The loop below walks the model's columns and reads the standard form's
  // objective, which is only the same list while the two have the same width.
  // They do today - to_standard_form splits rows, never columns - and if that
  // ever stops being true this reads a short list and concludes "integral" from
  // the coefficients it happened to see. Checking costs nothing and the failure
  // it guards against is a wrong answer, which is the expensive kind.
  bool integral_objective = options.objective_integrality &&
                            !integer_columns.empty() &&
                            sf.lp.num_cols() == model.num_cols() &&
                            sf.lp.c.size() == sz(model.num_cols());
  for (Int j = 0; j < model.num_cols() && integral_objective; ++j) {
    const double cj = sf.lp.c[sz(j)];
    if (cj == 0.0) continue;
    if (!is_integral[sz(j)] || std::fabs(cj - std::round(cj)) > 1e-9)
      integral_objective = false;
  }

  result.integral_objective = integral_objective;

  // A minimisation in standard form; the model's own sense is restored at the
  // end through objective_scale. Bounds compare in the standard direction.
  const bool maximizing = model.sense == ObjSense::kMaximize;
  auto to_model_objective = [&](double standard) {
    return sf.lp.objective_scale * standard + sf.lp.objective_offset;
  };

  double incumbent = kInf;  // best standard-form objective found
  std::vector<double> incumbent_x;

  // The value a node's bound has to reach before the node is worthless. Without
  // an integral objective that is the incumbent itself; with one it is a whole
  // unit lower, because no subtree can produce a value strictly between
  // incumbent - 1 and incumbent.
  auto cutoff = [&]() {
    if (incumbent >= kInf) return kInf;
    if (!integral_objective) return incumbent;
    // The unit is only there to be claimed if a unit is larger than the noise
    // in the numbers it is being subtracted from. A node bound is an LP
    // objective, carrying something like 1e-9 of relative rounding, so on an
    // instance whose objective runs to 1e6 that is 1e-3 in absolute terms - and
    // a node whose true bound is exactly incumbent - 1 can compute a shade
    // above it and be pruned, taking a strictly better answer with it.
    //
    // So the slack scales, and past the point where it swallows the unit the
    // rule turns itself off rather than guessing. This is the same lesson as
    // the two absolute 1e-9 tolerances elsewhere in this file, both of which
    // cost an answer: a tolerance on a quantity whose scale the model chooses
    // has to scale with it.
    const double slack = std::fmax(1e-6, 1e-9 * std::fabs(incumbent));
    if (slack >= 0.5) return incumbent;
    return incumbent - 1.0 + slack;
  };

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
  // What the cut loop leaves behind for the tree's root node to start from.
  std::vector<Int> root_basis;
  std::vector<VarStatus> root_basis_status;

  // `deadline` caps this one solve at a wall-clock time, on top of the whole
  // solve's budget. The cut loop needs it: a round has to be able to say "if
  // checking this costs more than the round saved, it did not pay".
  auto solve_node = [&](const Node& node, NodeSolution* out,
                        Int iteration_factor = -1, double deadline = kInf) {
    StandardLp lp = working;
    lp.lower = node.lower;
    lp.upper = node.upper;
    *out = NodeSolution{};
    result.relaxations_solved++;

    if (options.node_solver == BranchAndBoundOptions::NodeSolver::kSimplex) {
      SimplexOptions so = options.simplex;
      so.algorithm = SimplexOptions::Algorithm::kDual;
      const Int factor =
          iteration_factor >= 0 ? iteration_factor : options.node_iteration_factor;
      if (factor > 0) {
        so.max_iterations = std::max<Int>(200, factor * lp.num_rows());
      }
      // No single relaxation may outlive the budget for the whole solve. This
      // is not a refinement: without it the time limit only bounded the node
      // loop, and everything before that loop - the root relaxation and eight
      // rounds of cut separation, each its own LP - ran unbounded. 10teams
      // asked for 15 seconds and took 103.8, which is not a solver reporting a
      // time limit, it is a solver ignoring one.
      //
      // The budget has to be read immediately before each solve, and it used to
      // be read once here. Two things spend time between that read and the
      // solve it was meant for: the first-order seed below, which may take a
      // fifth of what is left, and the cold retry after it, which was handed a
      // copy of the same stale number and so got the whole limit a second time.
      // Three things each entitled to the full budget is not a budget. Measured
      // on the wider MIPLIB set at a 15 second limit, that is what the wall
      // clock was actually doing: 10teams 28.8s, acc-tight4 31.3s.
      auto budget_left = [&]() {
        const double whole = options.time_limit_seconds - elapsed();
        const double mine = deadline - elapsed();
        return std::max(0.0, std::fmin(whole, mine));
      };
      // A crossover basis has to outlive the solve it is passed to, so it is
      // declared here rather than inside the branch that fills it.
      CrossoverResult root_cross;
      if (static_cast<Int>(node.basic.size()) == lp.num_rows() &&
          !node.basis_status.empty()) {
        so.start_basic = &node.basic;
        so.start_status = &node.basis_status;
      } else if (options.root_crossover) {
        PdhgOptions po;
        po.tolerance = options.root_crossover_tolerance;
        po.gap_tolerance = options.root_crossover_tolerance;
        po.max_iterations = options.root_crossover_max_iterations;
        if (options.root_crossover_iterations_per_row > 0.0) {
          po.max_iterations = std::max<Int>(
              po.max_iterations,
              static_cast<Int>(options.root_crossover_iterations_per_row *
                               lp.num_rows()));
        }
        // An iteration cap is not a time bound on a large enough model, and
        // this is a seed for a solve that has its own budget. Same rule as the
        // relaxations: nothing here outlives the run it is part of.
        po.time_limit_seconds =
            std::fmax(0.0, options.root_crossover_time_share *
                               (options.time_limit_seconds - elapsed()));
        const PdhgResult seed = solve_pdhg(lp, po);
        if (!seed.x.empty()) {
          root_cross = crossover_basis(lp, seed.x, seed.y);
          if (root_cross.ok) {
            so.start_basic = &root_cross.basic;
            so.start_status = &root_cross.status;
            result.root_crossover_pushed = root_cross.pushed;
          }
        }
      }
      so.time_limit_seconds = budget_left();
      SimplexResult r = solve_lp(lp, so);
      // Same rule as everywhere else this appears: a starting basis may save
      // pivots and may do nothing, but it may not cost an answer.
      if (root_cross.ok && so.start_basic == &root_cross.basic &&
          r.status != SimplexStatus::kOptimal) {
        SimplexOptions cold = so;
        cold.start_basic = nullptr;
        cold.start_status = nullptr;
        cold.time_limit_seconds = budget_left();
        const SimplexResult again = solve_lp(lp, cold);
        if (again.status == SimplexStatus::kOptimal) r = again;
      } else if (root_cross.ok && r.started_warm) {
        result.root_crossover_used = true;
      }
      result.simplex_iterations += r.iterations;
      out->warm = r.started_warm;
      if (r.started_warm) result.warm_started_nodes++;
      out->optimal = r.status == SimplexStatus::kOptimal;
      out->proved_infeasible = r.status == SimplexStatus::kInfeasible;
      if (out->optimal) {
        out->x = r.x;
        out->y = r.y;
        out->basic = r.final_basic;
        out->basis_status = r.final_status;
        // The simplex stops on an exact optimality test, so the objective at
        // the vertex it stopped on is the relaxation's optimum and is a bound.
        out->bound = lp.standard_objective(out->x);
      }
      return out->optimal;
    }

    // The parent's point. This was being stored on every node and read by
    // nothing, so the option below claimed an inheritance that never happened;
    // the first-order path started every child from the origin.
    PdhgOptions po = options.relaxation;
    if (options.warm_start && node.warm_x.size() == sz(lp.num_cols())) {
      po.warm_x = &node.warm_x;
    }
    const PdhgResult p = solve_pdhg(lp, po);
    out->proved_infeasible = p.status == PdhgStatus::kPrimalInfeasible;
    out->x = p.x;

    // Where a first-order node's bound has to come from, and it is not from the
    // point the method returned.
    //
    // A first-order method stops on a tolerance. The point it hands back is
    // nearly feasible and nearly optimal, and a point that is feasible but a
    // little short of optimal has an objective ABOVE the relaxation's optimum.
    // Using that as a lower bound on the subtree over-estimates the bound, and
    // an over-estimated lower bound is exactly how a prune throws away the
    // answer - the failure this file already carries two write-ups of.
    //
    // What is a bound is the dual objective, q'y plus the bound term, by weak
    // duality: any dual feasible y gives a lower bound on every primal feasible
    // x, with no assumption that either is optimal. The condition is that y is
    // dual feasible, which is what dual_residual_inf measures. Where it is not
    // small this node has established nothing, and it is reported unsolved -
    // which the tree already handles by refusing to certify optimality - rather
    // than pruned on a number that is not a proof.
    //
    // The presolve session found the way this bites from the other side: a
    // stronger presolve moves the quantities the relative convergence test
    // divides by, so the same tolerance is a weaker requirement on the reduced
    // model, and the returned point can sit further above the optimum than the
    // tolerance suggests. On dfl001 that is 0.08% of the objective while every
    // violation is under 1e-6. The simplex is not exposed to it; this path is.
    const double dual_feasible_enough =
        1e-7 * std::fmax(1.0, std::fabs(p.residual.dual_objective));
    out->optimal = p.status == PdhgStatus::kOptimal &&
                   p.residual.dual_residual_inf <= dual_feasible_enough;
    if (out->optimal) out->bound = p.residual.dual_objective;
    return out->optimal;
  };
  // Which cut produced which appended row, so an invalid one can be named.
  const Int rows_before_cuts = working.num_rows();
  std::vector<std::string> cut_family;

  if (options.root_cuts && !integer_columns.empty()) {
    // Kept so the whole cut effort can be undone if it turns out not to have
    // paid. Measured in the standard form, where the objective is always a
    // minimisation and a valid cut can only raise the bound - the model-sense
    // numbers reported below move in whichever direction the model's own sense
    // implies, which is not what a threshold wants to read.
    const StandardLp uncut = working;
    double root_std_before = 0.0;
    double root_std_after = 0.0;
    bool have_root_std = false;

    // The basis carried between cut rounds. Without it every round is a cold
    // solve, and with root crossover in front of it every round is also a fresh
    // first-order seed - eight rounds, two solves each, sixteen seeds for one
    // root. Adding cuts does not invalidate a basis: the new rows arrive with
    // their own logicals, and a logical basic in its own row is exactly what
    // makes the extended basis nonsingular.
    std::vector<Int> carry_basic;
    std::vector<VarStatus> carry_status;
    // The cuts themselves, in the order they were appended, so a subset of them
    // can be rebuilt onto `uncut` below.
    std::vector<Cut> kept_cuts;
    const double cut_budget =
        options.root_cut_time_share * options.time_limit_seconds;
    auto carry_from = [&](const NodeSolution& solved) {
      if (static_cast<Int>(solved.basic.size()) == working.num_rows() &&
          !solved.basis_status.empty()) {
        carry_basic = solved.basic;
        carry_status = solved.basis_status;
      }
    };
    auto extend_carry = [&](Int rows_before_append) {
      if (carry_basic.empty()) return;
      const Int structural = working.num_cols();
      carry_status.resize(sz(structural + working.num_rows()), VarStatus::kAtLower);
      for (Int i = rows_before_append; i < working.num_rows(); ++i) {
        carry_basic.push_back(structural + i);
        carry_status[sz(structural + i)] = VarStatus::kBasic;
      }
    };

    for (Int round = 0; round < options.cut_rounds; ++round) {
      // Cut rounds are the expensive part of the root, and each is a full LP.
      // Stopping between rounds keeps whatever the earlier rounds bought.
      if (elapsed() > cut_budget) break;
      // Through the same solver the nodes use, because that is what makes the
      // tableau available - and Gomory cuts are read off the tableau.
      Node probe;
      probe.lower = working.lower;
      probe.upper = working.upper;
      probe.basic = carry_basic;
      probe.basis_status = carry_status;
      NodeSolution root_solve;
      const double solve_began = elapsed();
      if (!solve_node(probe, &root_solve)) break;
      const double solve_cost = elapsed() - solve_began;
      carry_from(root_solve);
      if (round == 0) {
        result.root_bound_before_cuts =
            sf.lp.objective_scale * root_solve.bound + sf.lp.objective_offset;
      }
      result.root_bound_after_cuts =
          sf.lp.objective_scale * root_solve.bound + sf.lp.objective_offset;
      root_std_after = root_solve.bound;
      if (!have_root_std) {
        root_std_before = root_std_after;
        have_root_std = true;
      }

      // Whether to cut at all is a question about what a round costs, and the
      // answer has to be measured on this instance rather than assumed. A round
      // is not one LP: it separates, appends, and re-solves to check the bound
      // did not fall, and the loop ends with a third solve to decide whether any
      // of it paid. So committing to a round commits to roughly three more root
      // relaxations.
      //
      // On an instance whose root takes four seconds of a fifteen second budget
      // that is the entire run, and the tree never starts. Measured over the 27
      // instances that ended with no solution at all: with cuts on, one of them
      // finds anything; with cuts off outright, four do. Cutting the time share
      // to a tenth changes nothing, which is what says the cost is the solves
      // and not the slice - a smaller slice still buys the first round, and the
      // first round is what commits to the rest.
      //
      // So the test is whether the round fits, not whether the slice is empty.
      // The root solve above is not part of the cost: the tree needs it anyway.
      if (elapsed() + 2.0 * solve_cost > cut_budget) break;

      CutOptions cut_options;
      cut_options.max_cuts = options.cuts_per_round;
      cut_options.cover_cuts = options.cover_cuts;
      cut_options.separate_only_before_row = rows_before_cuts;
      cut_options.mir_cuts = options.mir_cuts;
      std::vector<Cut> cuts =
          separate_cuts(working, is_integral, root_solve.x, cut_options);

      // Cover and MIR cuts come off the model's own rows; Gomory cuts come off
      // a combination of them, so they see things the others cannot. They are
      // ranked together rather than each family getting a quota, which is the
      // point of having one selector.
      cut_options.gomory_cuts =
          options.gomory_cuts && round < options.gomory_rounds;
      if (cut_options.gomory_cuts && !root_solve.basic.empty()) {
        std::vector<Cut> gomory = separate_gomory_cuts(
            working, is_integral, root_solve.basic, root_solve.basis_status,
            cut_options);
        result.gomory_cuts_added += static_cast<Int>(gomory.size());
        if (!gomory.empty()) {
          cuts.insert(cuts.end(), gomory.begin(), gomory.end());
          cuts = select_cuts(std::move(cuts), cut_options);
        }
      }
      if (cuts.empty()) break;
      // Adding a valid cut can only raise the bound. If it falls, at least one
      // of the cuts just added was not valid, and continuing would report a
      // bound that is not one. This does not catch a bad cut that happens to
      // raise the bound - nothing cheap does - so it is a safety net rather
      // than a proof, and the round limit above is the actual mitigation.
      const StandardLp before = working;
      const Int rows_before_append = before.num_rows();
      // The basis as it stood for `before`. extend_carry grows it to match the
      // appended rows, so a rollback has to put it back or the carry is one
      // basis for a relaxation of a different size - which solve_node then
      // rejects, sending the next root solve down the crossover path for a
      // fresh first-order seed and a cold simplex. That is the expensive thing
      // this loop is trying to stop doing.
      const std::vector<Int> carry_basic_before = carry_basic;
      const std::vector<VarStatus> carry_status_before = carry_status;
      auto roll_back = [&]() {
        working = before;
        carry_basic = carry_basic_before;
        carry_status = carry_status_before;
      };
      working = append_cuts(working, cuts);
      extend_carry(rows_before_append);
      Node recheck;
      recheck.lower = working.lower;
      recheck.upper = working.upper;
      recheck.basic = carry_basic;
      recheck.basis_status = carry_status;
      NodeSolution after;
      // The check gets a few times what the uncut root cost, and no more.
      //
      // What that measures is not how long the check takes but what the cuts
      // did to the relaxation, because it is the same relaxation plus their
      // rows. newdano is the case: its root solves in about a second, twenty
      // cuts turn it into five, and three of those are the whole fifteen second
      // budget - so the tree never starts, no heuristic ever runs, and the
      // instance ends with nothing. With cuts off it finds a solution in eight
      // nodes.
      //
      // A round that cannot re-solve its own relaxation in a small multiple of
      // what the relaxation used to cost has made every node below it that much
      // more expensive, and it is buying a bound for a tree that will not be
      // explored. Roll it back and stop cutting. That is also the safer
      // behaviour on its own terms: this solve is what checks the bound did not
      // fall, and cuts whose check did not finish used to be kept anyway.
      const double check_deadline =
          std::fmin(cut_budget, elapsed() + std::fmax(0.5, 3.0 * solve_cost));
      if (solve_node(recheck, &after, -1, check_deadline)) {
        carry_from(after);
        const double raised = after.bound;
        if (raised < root_solve.bound - 1e-6) {
          roll_back();
          result.cuts_rolled_back = true;
          break;
        }
      } else {
        roll_back();
        ++result.cut_rounds_abandoned;
        break;
      }
      for (const Cut& c : cuts) {
        cut_family.push_back(std::string(c.family) + " round " +
                             std::to_string(round + 1));
        kept_cuts.push_back(c);
      }
      result.cuts_added += static_cast<Int>(cuts.size());
    }

    // Did any of that move the bound? The final root solve is the one that
    // answers it, since the loop above measures before adding the last round's
    // cuts.
    if (have_root_std && result.cuts_added > 0) {
      Node final_probe;
      final_probe.lower = working.lower;
      final_probe.upper = working.upper;
      // The basis the rounds have been carrying. Without it this solve has none,
      // which sends it down the root-crossover path for a fresh first-order seed
      // and then a cold simplex - a second full root solve, at the end of a loop
      // that has just done several, on a relaxation one round of cuts away from
      // the one it already solved.
      final_probe.basic = carry_basic;
      final_probe.basis_status = carry_status;
      NodeSolution final_solve;
      const bool solved = solve_node(final_probe, &final_solve);
      if (solved) {
        root_std_after = final_solve.bound;
        result.root_bound_after_cuts =
            sf.lp.objective_scale * root_std_after + sf.lp.objective_offset;
      }
      result.root_bound_rise = (root_std_after - root_std_before) /
                               std::fmax(1.0, std::fabs(root_std_before));
      if (options.adaptive_cuts &&
          result.root_bound_rise < options.cut_bound_improvement) {
        // The cuts cost every node below here and bought nothing measurable.
        working = uncut;
        result.cuts_discarded = true;
        kept_cuts.clear();
        cut_family.clear();
      } else if (options.root_cut_filtering && solved &&
                 !kept_cuts.empty() &&
                 static_cast<Int>(kept_cuts.size()) ==
                     working.num_rows() - rows_before_cuts) {
        // Drop the cuts the root optimum does not sit on.
        //
        // A constraint that is slack at an optimal point is not holding it
        // there: remove it and the same point is still feasible and still
        // optimal, with the multiplier it never had. So the root bound is
        // unchanged by construction rather than by measurement - what changes
        // is that every node below solves a smaller LP.
        //
        // What it gives up is the deeper tree. Branching moves the relaxation,
        // and a cut that was slack at the root can bind once a variable is
        // pinned; a pool would keep those and re-add them where they bite.
        // There is no pool here, so this is the version of cut management that
        // fits: keep what did the work at the root, drop what only costs.
        std::vector<Cut> tight;
        std::vector<std::string> tight_family;
        for (std::size_t c = 0; c < kept_cuts.size(); ++c) {
          const Int row = rows_before_cuts + static_cast<Int>(c);
          double activity = 0.0;
          for (Int e = working.k.row_begin(row); e < working.k.row_end(row); ++e) {
            activity +=
                working.k.value()[sz(e)] * final_solve.x[sz(working.k.index()[sz(e)])];
          }
          // Cuts join as >= rows, so slack is activity - q.
          const double q = working.q[sz(row)];
          const double scale = std::fmax(1.0, std::fabs(q));
          if (activity - q <= 1e-7 * scale) {
            tight.push_back(kept_cuts[c]);
            if (c < cut_family.size()) tight_family.push_back(cut_family[c]);
          }
        }
        result.cuts_dropped_slack =
            static_cast<Int>(kept_cuts.size() - tight.size());
        if (result.cuts_dropped_slack > 0) {
          working = append_cuts(uncut, tight);
          kept_cuts.swap(tight);
          cut_family.swap(tight_family);
          // The rows moved, so the basis that was carried no longer describes
          // this relaxation. Dropping it is right; keeping it would be wrong.
          carry_basic.clear();
          carry_status.clear();
        }
      }
    }
    root_basis = carry_basic;
    root_basis_status = carry_status;
  }

  // Debug-solution tracking. `contains` asks whether the reference solution
  // lies inside a node's box; `report` names the first prune that discards a
  // box holding it, which is the prune that lost the answer.
  bool debug_lost = false;
  auto contains = [&](const std::vector<double>& lo,
                      const std::vector<double>& hi) {
    const std::vector<double>* d = options.debug_solution;
    if (d == nullptr) return false;
    const Int n = std::min<Int>(working.num_cols(), static_cast<Int>(d->size()));
    for (Int j = 0; j < n; ++j) {
      const double v = (*d)[sz(j)];
      if (v < lo[sz(j)] - 1e-6 || v > hi[sz(j)] + 1e-6) return false;
    }
    return true;
  };
  auto report = [&](const std::vector<double>& lo, const std::vector<double>& hi,
                    const char* why, double extra) {
    if (debug_lost || !contains(lo, hi)) return;
    debug_lost = true;
    std::printf("DEBUG SOLUTION LOST: %s (value %.10g)\n", why, extra);
  };

  // The cut loop is finished, so `working` is the model the tree will search.
  // If the reference solution does not satisfy it, a cut that was kept is
  // invalid, and everything after this is searching the wrong problem.
  if (options.debug_solution != nullptr) {
    std::vector<double> ref = *options.debug_solution;
    ref.resize(sz(working.num_cols()), 0.0);
    std::vector<double> scratch;
    double two = 0.0, inf = 0.0;
    working.primal_residual(ref, &scratch, &two, &inf);
    std::printf("after cuts: the reference solution violates the model by %.6e%s\n",
                inf, inf > 1e-6 ? "   <- A KEPT CUT IS INVALID" : "");
    if (inf > 1e-6) {
      for (Int i = rows_before_cuts; i < working.num_rows(); ++i) {
        double act = 0.0;
        for (Int e = working.k.row_begin(i); e < working.k.row_end(i); ++e)
          act += working.k.value()[sz(e)] * ref[sz(working.k.index()[sz(e)])];
        if (working.q[sz(i)] - act > 1e-6) {
          const Int which = i - rows_before_cuts;
          std::printf("  the offending row came from %s\n",
                      which < static_cast<Int>(cut_family.size())
                          ? cut_family[sz(which)].c_str() : "an unknown cut");
          break;
        }
      }
    }
  }

  std::vector<Node> stack;
  Node root;
  root.lower = working.lower;
  root.upper = working.upper;
  // The basis the cut loop ended on, when it left one that fits the relaxation
  // the tree is about to search. Without it the root node has no basis and goes
  // down the crossover path - a first-order seed and a cold simplex - to solve
  // a relaxation the cut loop has just solved, two or three times over.
  if (static_cast<Int>(root_basis.size()) == working.num_rows() &&
      !root_basis_status.empty()) {
    root.basic = root_basis;
    root.basis_status = root_basis_status;
  }
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

  // Feasibility pump. See branch_and_bound.hpp for why this exists at all: on
  // most of the wider MIPLIB set neither rounding nor diving finds anything, and
  // a tree with no incumbent has nothing to prune against.
  //
  // Two points are pulled together. `target` is integral and need not satisfy
  // the rows; `x` satisfies the rows and need not be integral. Each round
  // refreshes the first by rounding the second, then re-solves the relaxation
  // with the objective replaced by the distance to it.
  auto run_feasibility_pump = [&](const std::vector<double>& start_x,
                                  double deadline) {
    if (integer_columns.empty() || start_x.empty()) return false;
    const Int n = working.num_cols();

    StandardLp pump = working;  // same rows, same bounds; only c changes
    std::vector<double> x = start_x;
    std::vector<double> target(sz(n), 0.0);
    std::vector<double> previous;
    std::vector<Int> basic;
    std::vector<VarStatus> basis_status;
    bool have_basis = false;
    // Fixed seed: a heuristic that finds a solution only on some runs is not a
    // result anyone can reproduce.
    std::mt19937 rng(20260830u);
    Int restarts = 0;
    double objective_weight = options.pump_objective_weight;

    for (Int round = 0; round < options.pump_max_rounds; ++round) {
      if (elapsed() > deadline) break;
      result.pump_rounds++;

      // Round the integers; leave the continuous part where the relaxation put
      // it, since it is already consistent with the rows.
      target = x;
      for (const Int j : integer_columns) {
        const double rounded = std::round(x[sz(j)]);
        target[sz(j)] = std::fmin(std::fmax(rounded, working.lower[sz(j)]),
                                  working.upper[sz(j)]);
      }
      if (try_incumbent(target)) {
        result.pump_successes++;
        return true;
      }

      // The same rounding twice means the distance LP has stopped moving.
      // Flipping the columns the rounding was least sure about is what breaks
      // the cycle; taking the least sure ones rather than random ones is what
      // makes the escape purposeful.
      if (!previous.empty()) {
        bool identical = true;
        for (const Int j : integer_columns) {
          if (std::fabs(target[sz(j)] - previous[sz(j)]) > 0.5) {
            identical = false;
            break;
          }
        }
        if (identical) {
          if (++restarts > options.pump_max_restarts) break;
          result.pump_restarts++;
          std::vector<std::pair<double, Int>> doubt;
          doubt.reserve(integer_columns.size());
          for (const Int j : integer_columns) {
            doubt.push_back({std::fabs(x[sz(j)] - target[sz(j)]), j});
          }
          std::sort(doubt.begin(), doubt.end(),
                    [](const std::pair<double, Int>& a,
                       const std::pair<double, Int>& b) { return a.first > b.first; });
          const std::size_t half = std::max<std::size_t>(1, doubt.size() / 20);
          std::uniform_int_distribution<std::size_t> how_many(half, 3 * half);
          const std::size_t flips = std::min(how_many(rng), doubt.size());
          for (std::size_t k = 0; k < flips; ++k) {
            const std::size_t j = sz(doubt[k].second);
            const double away = (x[j] > target[j]) ? 1.0 : -1.0;
            target[j] = std::fmin(std::fmax(target[j] + away, working.lower[j]),
                                  working.upper[j]);
          }
        }
      }
      previous = target;

      // Distance to `target`, linearised. An integer column whose target sits
      // on one of its own bounds contributes its distance from that bound, which
      // is linear. One landing strictly inside its range would need an auxiliary
      // variable per column to write |x_j - t_j| as a linear term, which is the
      // general-integer form of the pump and is not built here - those columns
      // get no term, so the pump pulls on the binaries and lets the rest follow.
      pump.c.assign(sz(n), 0.0);
      Int pulled = 0;
      for (const Int j : integer_columns) {
        const double lower = working.lower[sz(j)];
        const double upper = working.upper[sz(j)];
        if (target[sz(j)] <= lower + 0.5 && lower > -kInf) {
          pump.c[sz(j)] = 1.0;
          ++pulled;
        } else if (target[sz(j)] >= upper - 0.5 && upper < kInf) {
          pump.c[sz(j)] = -1.0;
          ++pulled;
        } else if (options.pump_interior_terms) {
          // The target sits strictly inside the column's range, where
          // |x_j - t_j| needs an auxiliary variable to be written linearly. The
          // general-integer pump does that; this one used to give the column no
          // term at all, so on a model whose integers are not binaries the pump
          // was pulling on nothing and re-solving the same LP until it gave up.
          //
          // A subgradient of |x_j - t_j| at the current point costs no variable
          // and points the right way: push the column back toward the target
          // from whichever side it is on. It is only valid near the current
          // point, which is all that is needed - the pump re-solves and
          // re-rounds every round, so the linearisation is rebuilt each time
          // from wherever the relaxation has moved to.
          //
          // The direction has to have a bound to stop at, or the objective is
          // unbounded and the round returns nothing.
          const double away = x[sz(j)] - target[sz(j)];
          if (away > 1e-9 && lower > -kInf) {
            pump.c[sz(j)] = 1.0;
            ++pulled;
          } else if (away < -1e-9 && upper < kInf) {
            pump.c[sz(j)] = -1.0;
            ++pulled;
          }
        }
      }
      if (pulled == 0) break;

      // Mix in a decaying share of the real objective. Both terms are
      // normalised first, because one is a count of columns and the other is
      // in the model's own units - adding them unnormalised would let whichever
      // has the larger units decide everything.
      if (objective_weight > 0.0) {
        double distance_norm = 0.0;
        double cost_norm = 0.0;
        for (Int j = 0; j < n; ++j) {
          distance_norm += pump.c[sz(j)] * pump.c[sz(j)];
          cost_norm += working.c[sz(j)] * working.c[sz(j)];
        }
        distance_norm = std::sqrt(distance_norm);
        cost_norm = std::sqrt(cost_norm);
        if (distance_norm > 0.0 && cost_norm > 0.0) {
          const double a = objective_weight;
          for (Int j = 0; j < n; ++j) {
            pump.c[sz(j)] = (1.0 - a) * pump.c[sz(j)] / distance_norm +
                            a * working.c[sz(j)] / cost_norm;
          }
        }
        objective_weight *= options.pump_objective_decay;
      }

      SimplexOptions so = options.simplex;
      so.algorithm = SimplexOptions::Algorithm::kPrimal;
      so.time_limit_seconds = std::max(0.0, deadline - elapsed());
      if (options.pump_iteration_factor > 0) {
        so.max_iterations =
            std::max<Int>(200, options.pump_iteration_factor * pump.num_rows());
      }
      // Only the objective moved, so the previous basis is still primal
      // feasible and the primal simplex re-optimises from it in a few pivots.
      if (have_basis && static_cast<Int>(basic.size()) == pump.num_rows()) {
        so.start_basic = &basic;
        so.start_status = &basis_status;
      }
      const SimplexResult r = solve_lp(pump, so);
      result.simplex_iterations += r.iterations;
      if (r.status != SimplexStatus::kOptimal || r.x.empty()) break;
      basic = r.final_basic;
      basis_status = r.final_status;
      have_basis = !basic.empty();

      // Converged: the relaxation point is already integral, so the two points
      // have met.
      bool integral = true;
      for (const Int j : integer_columns) {
        if (fractionality(r.x[sz(j)]) > options.integrality_tolerance) {
          integral = false;
          break;
        }
      }
      x = r.x;
      if (integral && try_incumbent(x)) {
        result.pump_successes++;
        return true;
      }
    }
    return false;
  };

  // LP-guided diving. See branch_and_bound.hpp for why this exists beside the
  // fix-and-propagate dive: that one reads every decision off a single
  // relaxation and never re-solves, so the further it goes the staler its guide
  // is. This one re-solves after each decision, which is affordable because a
  // decision is one bound on one variable - the same change a branching child
  // makes, and the same warm start.
  auto run_lp_dive = [&](const Node& from, const NodeSolution& relaxation,
                         double deadline) {
    if (integer_columns.empty() || relaxation.x.empty()) return false;
    if (static_cast<Int>(relaxation.basic.size()) != working.num_rows()) return false;
    ++result.lp_dives_run;

    std::vector<double> lo = from.lower;
    std::vector<double> hi = from.upper;
    std::vector<double> x = relaxation.x;
    std::vector<Int> basic = relaxation.basic;
    std::vector<VarStatus> basis_status = relaxation.basis_status;

    for (Int step = 0; step < options.lp_dive_max_steps; ++step) {
      if (elapsed() > deadline) return false;

      // Nearest an integer goes first: it is the smallest disturbance on the
      // table, so it is the decision least likely to make the relaxation
      // infeasible, and it is the one the relaxation is closest to making
      // already.
      Int pick = -1;
      double least = kInf;
      for (const Int j : integer_columns) {
        const double f = fractionality(x[sz(j)]);
        if (f <= options.integrality_tolerance) continue;
        if (f < least) {
          least = f;
          pick = j;
        }
      }
      if (pick < 0) {
        // Nothing fractional is left, so this relaxation is already a feasible
        // point of the integer problem. try_incumbent re-checks it against the
        // rows rather than trusting the dive.
        std::vector<double> candidate = x;
        for (const Int j : integer_columns)
          candidate[sz(j)] = std::round(candidate[sz(j)]);
        if (try_incumbent(candidate)) {
          ++result.lp_dive_successes;
          return true;
        }
        return false;
      }

      const double value = x[sz(pick)];
      const double nearest = std::round(value);
      bool advanced = false;
      for (int attempt = 0; attempt < 2 && !advanced; ++attempt) {
        // The rounded side first, then the other one. Bound, do not fix: the
        // rounded side is a branching child, and it leaves the rest of the
        // column's range available to the relaxation.
        const bool go_up = (attempt == 0) ? (nearest > value) : (nearest <= value);
        std::vector<double> trial_lo = lo;
        std::vector<double> trial_hi = hi;
        if (go_up) {
          trial_lo[sz(pick)] = std::fmax(trial_lo[sz(pick)], std::ceil(value));
        } else {
          trial_hi[sz(pick)] = std::fmin(trial_hi[sz(pick)], std::floor(value));
        }
        if (trial_lo[sz(pick)] > trial_hi[sz(pick)]) continue;
        if (options.node_propagation &&
            !propagate_bounds(working, is_integral, &trial_lo, &trial_hi,
                              options.node_propagation_rounds)) {
          continue;
        }

        Node probe;
        probe.lower = trial_lo;
        probe.upper = trial_hi;
        probe.basic = basic;
        probe.basis_status = basis_status;
        NodeSolution solved;
        ++result.lp_dive_steps;
        if (!solve_node(probe, &solved, options.lp_dive_iteration_factor)) continue;
        // A dive whose relaxation is already worse than the incumbent cannot
        // reach anything worth having, and carrying on would spend the slice on
        // a point that would be rejected at the end.
        if (solved.bound >= cutoff()) return false;

        lo.swap(trial_lo);
        hi.swap(trial_hi);
        x = solved.x;
        basic = solved.basic;
        basis_status = solved.basis_status;
        advanced = true;
      }
      // Neither side is available, and this dive has no deeper backtracking to
      // offer. Another one from a different relaxation is cheaper than
      // unwinding this one.
      if (!advanced) return false;
    }
    return false;
  };

  // RINS. See branch_and_bound.hpp for the argument; the mechanics are that the
  // columns the incumbent and this node's relaxation agree on get fixed, and
  // what is left is handed to this same solver as a MIP in its own right.
  //
  // Recursion is bounded by switching RINS off in the child. Everything else is
  // left on, because a sub-MIP is a different problem and the strategies that
  // suit the parent need not suit it - which is Rothberg's point about why
  // fixing changes the nature of the problem rather than only its size.
  auto run_rins = [&](const std::vector<double>& node_x, double deadline) {
    if (incumbent_x.empty() || node_x.empty() || integer_columns.empty())
      return false;

    Model sub = model;
    Int fixed = 0;
    for (const Int j : integer_columns) {
      const double a = incumbent_x[sz(j)];
      const double b = node_x[sz(j)];
      if (fractionality(b) > options.integrality_tolerance) continue;
      if (std::fabs(a - b) > 0.5) continue;
      const double v = std::round(a);
      // Only inside the column's own bounds; the incumbent is feasible for the
      // model, so this always is, but the model's bounds are what the sub-MIP
      // will be validated against.
      if (v < sub.col_lower[sz(j)] - 1e-9 || v > sub.col_upper[sz(j)] + 1e-9)
        continue;
      sub.col_lower[sz(j)] = v;
      sub.col_upper[sz(j)] = v;
      ++fixed;
    }
    const double share = static_cast<double>(fixed) /
                         static_cast<double>(integer_columns.size());
    if (share < options.rins_min_fixed_fraction) return false;

    ++result.rins_run;
    result.rins_fixed += fixed;

    BranchAndBoundOptions so = options;
    so.rins = false;  // the recursion stops here
    so.debug_solution = nullptr;
    so.node_limit = options.rins_node_limit;
    so.time_limit_seconds = std::fmax(0.0, deadline - elapsed());
    so.verbose = false;
    // The sub-MIP is small by construction; a cut loop on top of it is the
    // fixed cost this whole file has just been taught to avoid.
    so.root_cuts = false;
    so.root_crossover = false;
    if (so.time_limit_seconds <= 0.0) return false;

    const BranchAndBoundResult sr = solve_milp(sub, so);
    if (sr.x.empty()) return false;
    // Through the same gate every other heuristic goes through: the point is
    // re-checked against the rows and the bounds of the problem being solved,
    // not trusted because a solver returned it.
    if (try_incumbent(sr.x)) {
      ++result.rins_successes;
      return true;
    }
    return false;
  };

  bool cuts_reverted = false;
  // Whether a plunge is in progress, and how deep it has gone. A plunge ends
  // when a node produces no children - pruned, infeasible, integral - or when
  // it has gone deeper than the limit, whichever comes first.
  bool plunging = false;
  Int plunge_depth = 0;
  MilpStatus status = MilpStatus::kInfeasible;
  double pump_time_spent = 0.0;
  double lp_dive_time_spent = 0.0;
  double rins_time_spent = 0.0;

  while (!stack.empty()) {
    if (result.nodes >= options.node_limit) {
      status = MilpStatus::kNodeLimit;
      break;
    }
    if (elapsed() > options.time_limit_seconds) {
      status = MilpStatus::kTimeLimit;
      break;
    }

    // Which open node to take next.
    //
    // Depth first takes the last one pushed, which is the child just created,
    // and that is the right thing while a plunge is running: the child differs
    // from its parent in one bound, so the parent's basis re-optimises it in a
    // handful of pivots, and diving is what finds feasible points.
    //
    // It is the wrong thing between plunges. The open list under depth first
    // always holds a node whose parent bound is the root's, so the smallest
    // bound over open nodes - which is the only lower bound the solver can
    // report - does not move until the tree is nearly exhausted. Depth first
    // does not merely improve the dual bound slowly; it disregards it.
    //
    // So: plunge, and when the plunge ends take the node with the best estimate
    // of what lies below it rather than the deepest one. That is SCIP's default
    // node selector.
    std::size_t take = stack.size() - 1;
    if (options.node_selection == BranchAndBoundOptions::NodeSelection::kBestEstimate &&
        !plunging) {
      double best = kInf;
      for (std::size_t i = 0; i < stack.size(); ++i) {
        // Ties, and nodes whose estimate was never set, fall back to the bound.
        const double key = std::isfinite(stack[i].estimate) ? stack[i].estimate
                                                            : stack[i].parent_bound;
        if (key < best) {
          best = key;
          take = i;
        }
      }
      ++result.best_estimate_jumps;
      plunge_depth = 0;
    }
    if (take != stack.size() - 1) std::swap(stack[take], stack.back());
    Node node = std::move(stack.back());
    stack.pop_back();
    result.nodes++;
    result.max_depth = std::max(result.max_depth, node.depth);
    // Every path out of this iteration that does not push children ends the
    // plunge; the one that does re-arms it below.
    plunging = false;

    // Nothing below this node can beat the incumbent.
    if (node.parent_bound >= cutoff()) {
      report(node.lower, node.upper, "pruned on the parent bound", node.parent_bound);
      continue;
    }

    if (provably_infeasible(working, node.lower, node.upper)) {
      report(node.lower, node.upper, "declared infeasible by interval arithmetic", 0.0);
      result.nodes_proved_infeasible++;
      continue;
    }

    NodeSolution relaxation;
    if (!solve_node(node, &relaxation) && relaxation.proved_infeasible) {
      // Proved empty, so nothing below this node exists. Safe to prune.
      result.nodes_proved_infeasible++;
      continue;
    }
    if (!relaxation.optimal) {
      // A root that will not solve with the cuts on it should be tried without
      // them before the run is given up on.
      //
      // neos-3046615-murg is the case. 105 cuts take its root bound from 192 to
      // 288, and then the relaxation carrying them does not converge - so the
      // solver reported "relaxation failed" after 0.104 seconds of a fifteen
      // second budget, having spent all of it on a bound for a tree it then
      // declined to search. The cuts are what made the relaxation hard; the
      // relaxation without them is the one the tree was always entitled to.
      //
      // Same rule as the crossover basis two hundred lines up, and as the seeded
      // simplex in the LP path: a thing that is meant to help may cost time and
      // may do nothing, but it may not cost the answer. Once only, so a root
      // that will not solve either way still terminates.
      if (node.depth == 0 && !cuts_reverted && working.num_rows() > rows_before_cuts) {
        cuts_reverted = true;
        result.cuts_reverted_after_root_failure = true;
        working = sf.lp;  // append_cuts leaves bounds alone, so this is the
                          // same relaxation with the cut rows taken back off
        Node again;
        again.lower = working.lower;
        again.upper = working.upper;
        stack.push_back(std::move(again));
        continue;
      }
      // Treat a node whose relaxation did not solve as pruned, and say so, since
      // silently dropping subtrees would make the reported bound a lie.
      result.nodes_relaxation_failed++;
      continue;
    }

    // The bound the node's own solver established, not the objective at
    // whatever point it returned. Those are the same number under the simplex
    // and are not under a first-order method; see solve_node.
    const double node_objective = relaxation.bound;

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

    if (node_objective >= cutoff()) {
      report(node.lower, node.upper, "pruned on its own relaxation bound",
             node_objective);
      continue;  // bound prune
    }

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

    // Then the pump, at the root and only while there is still nothing to prune
    // against. It is the expensive heuristic and the only one that finds
    // anything on most of the wider set, so it goes first and gets a slice of
    // the budget rather than a node's worth.
    //
    // It runs again deeper in the tree while there is still nothing, because a
    // pump started from a different relaxation rounds differently and lands
    // somewhere else. The share of the budget is cumulative across all of them,
    // so retrying costs no more in total than one long attempt.
    if (options.feasibility_pump && incumbent_x.empty() &&
        (node.depth == 0 ||
         (options.pump_retry_nodes > 0 && result.nodes % options.pump_retry_nodes == 0))) {
      const double budget = options.pump_time_share * options.time_limit_seconds;
      const double left = budget - pump_time_spent;
      if (left > 0.0) {
        const double began = elapsed();
        const double deadline = std::fmin(options.time_limit_seconds, began + left);
        if (run_feasibility_pump(relaxation.x, deadline)) {
          result.heuristic_successes++;
        }
        pump_time_spent += elapsed() - began;
      }
    }

    // Then the LP-guided dive, on the same gate as the other two: it is for
    // finding a first point, and once there is one to prune against the tree is
    // the better use of the time.
    if (options.lp_diving && incumbent_x.empty() &&
        (node.depth == 0 ||
         (options.lp_dive_retry_nodes > 0 &&
          result.nodes % options.lp_dive_retry_nodes == 0))) {
      const double budget = options.lp_dive_time_share * options.time_limit_seconds;
      const double left = budget - lp_dive_time_spent;
      if (left > 0.0) {
        const double began = elapsed();
        const double deadline = std::fmin(options.time_limit_seconds, began + left);
        if (run_lp_dive(node, relaxation, deadline)) result.heuristic_successes++;
        lp_dive_time_spent += elapsed() - began;
      }
    }

    // RINS, which is the other way round from all of the above: it needs an
    // incumbent and makes it better, where they need nothing and find the first
    // one.
    if (options.rins && !incumbent_x.empty() && options.rins_nodes > 0 &&
        result.nodes % options.rins_nodes == 0) {
      const double budget = options.rins_time_share * options.time_limit_seconds;
      const double left = budget - rins_time_spent;
      if (left > 0.0) {
        const double began = elapsed();
        if (run_rins(relaxation.x,
                     std::fmin(options.time_limit_seconds, began + left))) {
          result.heuristic_successes++;
        }
        rins_time_spent += elapsed() - began;
      }
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
        // Through the same solver the nodes use, and warm started from this
        // node's own basis - the dive has only tightened bounds, so that basis
        // is still dual feasible for it.
        // No warm start, and a short leash. The dive has pinned every integer
        // column, so this LP barely resembles the one whose basis is to hand -
        // handing that basis over is worse than starting clean, and on neos5 it
        // was catastrophically worse: 200,000 iterations and 9,970 anti-cycling
        // switches on a 63-row problem, against 7 iterations for an ordinary
        // warm-started child. It is a heuristic. If it cannot find a point
        // quickly it should not be allowed to spend the solve's whole budget
        // failing to.
        Node dive_node;
        dive_node.lower = lo;
        dive_node.upper = hi;
        dive_node.basic = relaxation.basic;
        dive_node.basis_status = relaxation.basis_status;
        NodeSolution dive_result;
        solve_node(dive_node, &dive_result, options.dive_iteration_factor);
        if (dive_result.optimal) {
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
      // The product score, from whatever the pseudocosts currently say.
      auto score_of = [](double down, double up) {
        return std::fmax(down, 1e-6) * std::fmax(up, 1e-6) + 1e-6 * (down + up);
      };

      struct Candidate {
        Int column;
        double value;
        double down_frac;
        double up_frac;
        double score;
        bool reliable;
      };
      std::vector<Candidate> candidates;
      for (const Int j : integer_columns) {
        const double v = relaxation.x[sz(j)];
        if (fractionality(v) <= options.integrality_tolerance) continue;
        const double down_frac = v - std::floor(v);
        const double up_frac = std::ceil(v) - v;
        // An unmeasured direction gets an optimistic unit cost, so every
        // variable is tried once before the averages decide anything.
        const double dc = pseudo_down_n[sz(j)] ? pseudo_down[sz(j)] : 1.0;
        const double uc = pseudo_up_n[sz(j)] ? pseudo_up[sz(j)] : 1.0;
        const bool reliable =
            std::min(pseudo_down_n[sz(j)], pseudo_up_n[sz(j)]) >=
            options.reliability_threshold;
        candidates.push_back({j, v, down_frac, up_frac,
                              score_of(dc * down_frac, uc * up_frac), reliable});
      }

      // Strong branch the unreliable candidates, best-first, up to the budget.
      // Each probe is a bounded dual simplex from this node's own basis, so it
      // is a warm start away from the parent and cheap.
      const bool strong_here =
          options.reliability_branching &&
          (options.strong_branch_max_depth < 0 ||
           node.depth <= options.strong_branch_max_depth);
      if (strong_here && !candidates.empty()) {
        std::vector<Candidate*> unreliable;
        for (Candidate& c : candidates)
          if (!c.reliable) unreliable.push_back(&c);
        std::sort(unreliable.begin(), unreliable.end(),
                  [](const Candidate* a, const Candidate* b) {
                    return a->score > b->score;
                  });
        const std::size_t budget =
            std::min<std::size_t>(unreliable.size(),
                                  sz(std::max<Int>(0, options.strong_branch_candidates)));
        for (std::size_t k = 0; k < budget; ++k) {
          Candidate& c = *unreliable[k];
          double delta[2] = {0.0, 0.0};
          bool measured[2] = {false, false};
          bool pruned = false;
          for (int side = 0; side < 2 && !pruned; ++side) {
            Node probe;
            probe.lower = node.lower;
            probe.upper = node.upper;
            probe.basic = relaxation.basic;
            probe.basis_status = relaxation.basis_status;
            if (side == 0) {
              probe.upper[sz(c.column)] = std::floor(c.value);
            } else {
              probe.lower[sz(c.column)] = std::ceil(c.value);
            }
            if (probe.lower[sz(c.column)] > probe.upper[sz(c.column)]) continue;
            NodeSolution probe_solve;
            ++result.strong_branch_probes;
            if (solve_node(probe, &probe_solve,
                           options.strong_branch_iteration_factor)) {
              // The probe's bound against this node's, both established rather
            // than read off a returned point. Identical under the simplex.
            delta[side] = std::fmax(0.0, probe_solve.bound - node_objective);
              measured[side] = true;
            } else if (probe_solve.proved_infeasible) {
              // One side is empty, so this variable settles a whole subtree.
              // Branching here next is the best thing available.
              ++result.strong_branch_prunes;
              pruned = true;
            }
          }
          if (pruned) {
            c.score = kInf;
            continue;
          }
          // A strong branch is a measurement, so it feeds the pseudocost the
          // same way a real branch would - the unit cost per unit of
          // fractionality moved.
          // A probe that ran out of iterations has not said the bound change is
          // small, only that we did not wait for it. Scoring it as zero is how
          // an unconverged probe turns a good candidate into a rejected one -
          // it took gt2 from 783 nodes to 24,585. Where the probe did not
          // finish, fall back to what the pseudocost already believed.
          const double fallback_down =
              (pseudo_down_n[sz(c.column)] ? pseudo_down[sz(c.column)] : 1.0) *
              c.down_frac;
          const double fallback_up =
              (pseudo_up_n[sz(c.column)] ? pseudo_up[sz(c.column)] : 1.0) * c.up_frac;

          if (measured[0] && c.down_frac > 1e-9 && delta[0] > 0.0) {
            const Int n = pseudo_down_n[sz(c.column)];
            pseudo_down[sz(c.column)] =
                (pseudo_down[sz(c.column)] * static_cast<double>(n) +
                 delta[0] / c.down_frac) / static_cast<double>(n + 1);
            pseudo_down_n[sz(c.column)] = n + 1;
          }
          if (measured[1] && c.up_frac > 1e-9 && delta[1] > 0.0) {
            const Int n = pseudo_up_n[sz(c.column)];
            pseudo_up[sz(c.column)] =
                (pseudo_up[sz(c.column)] * static_cast<double>(n) +
                 delta[1] / c.up_frac) / static_cast<double>(n + 1);
            pseudo_up_n[sz(c.column)] = n + 1;
          }
          c.score = score_of(measured[0] ? delta[0] : fallback_down,
                             measured[1] ? delta[1] : fallback_up);
        }
      }

      double best = -1.0;
      for (const Candidate& c : candidates) {
        if (c.score > best) {
          best = c.score;
          branch_column = c.column;
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

    // Forrest's estimate of the best integer objective obtainable below here:
    // this node's relaxation value, plus for every column still fractional the
    // cheaper of its two roundings priced by the pseudocost that says what a
    // unit of that rounding has historically cost.
    //
    //   e = z + sum_j min( P_j^-  (x_j - floor x_j),  P_j^+  (ceil x_j - x_j) )
    //
    // Forrest, Hirst and Tomlin; it is what SCIP's default node selector ranks
    // on. It is not a bound and must never be used as one - it is a guess at
    // where a subtree ends up, which is what makes it the right thing to sort
    // by when the question is "which open node is worth going to next" and the
    // wrong thing entirely when the question is "can this subtree be discarded".
    double estimate_base = node_objective;
    double branch_share = 0.0;
    for (const Int j : integer_columns) {
      const double v = relaxation.x[sz(j)];
      if (fractionality(v) <= options.integrality_tolerance) continue;
      const double down = (v - std::floor(v)) *
                          (pseudo_down_n[sz(j)] ? pseudo_down[sz(j)] : 1.0);
      const double up = (std::ceil(v) - v) *
                        (pseudo_up_n[sz(j)] ? pseudo_up[sz(j)] : 1.0);
      const double cheaper = std::fmin(down, up);
      estimate_base += cheaper;
      if (j == branch_column) branch_share = cheaper;
    }

    // Reduced-cost fixing, for the children only. This node's bound plus what
    // it costs to move a variable off its bound is a lower bound on anything
    // below here that moves it; where that exceeds the incumbent, the rest of
    // the range is not worth carrying down.
    //
    // Which bound a variable sits at is read from the solution rather than from
    // the basis status, because the status is in the simplex's own indexing -
    // structurals and logicals together - and this only concerns structurals.
    std::vector<double> child_lower = node.lower;
    std::vector<double> child_upper = node.upper;
    if (options.reduced_cost_fixing && incumbent < kInf &&
        static_cast<Int>(relaxation.y.size()) == working.num_rows()) {
      // The same cutoff the prunes use, so an integral objective tightens every
      // bound this derives by a whole unit as well.
      const double gap = cutoff() - node_objective;
      if (gap >= 0.0 && std::isfinite(gap)) {
        std::vector<double> d = working.c;
        for (Int i = 0; i < working.num_rows(); ++i) {
          const double yi = relaxation.y[sz(i)];
          if (yi == 0.0) continue;
          for (Int e = working.k.row_begin(i); e < working.k.row_end(i); ++e) {
            d[sz(working.k.index()[sz(e)])] -= working.k.value()[sz(e)] * yi;
          }
        }
        const double at_bound = 1e-7;
        for (Int j = 0; j < working.num_cols(); ++j) {
          const double lo = child_lower[sz(j)];
          const double hi = child_upper[sz(j)];
          if (hi - lo <= at_bound) continue;  // already fixed
          const double xj = relaxation.x[sz(j)];
          const double dj = d[sz(j)];
          const bool integral = j < static_cast<Int>(is_integral.size()) && is_integral[sz(j)];
          if (std::isfinite(lo) && xj - lo <= at_bound && dj > 1e-9) {
            double bound = lo + gap / dj;
            if (integral) bound = std::floor(bound + 1e-9);
            if (bound < hi) {
              child_upper[sz(j)] = std::fmax(bound, lo);
              ++result.reduced_cost_tightenings;
              if (child_upper[sz(j)] - lo <= at_bound) ++result.reduced_cost_fixings;
            }
          } else if (std::isfinite(hi) && hi - xj <= at_bound && dj < -1e-9) {
            double bound = hi - gap / (-dj);
            if (integral) bound = std::ceil(bound - 1e-9);
            if (bound > lo) {
              child_lower[sz(j)] = std::fmin(bound, hi);
              ++result.reduced_cost_tightenings;
              if (hi - child_lower[sz(j)] <= at_bound) ++result.reduced_cost_fixings;
            }
          }
        }
      }
    }

    // Branching itself always excludes the solution from one of the two
    // children, which is not a loss. Reduced-cost fixing narrowing the box out
    // from under it is.
    if (contains(node.lower, node.upper) && !contains(child_lower, child_upper)) {
      report(node.lower, node.upper, "reduced-cost fixing excluded it", incumbent);
    }

    // Depth first within a plunge, and the child nearer the relaxation value is
    // explored first, because it is the likelier place to find a feasible point
    // early.
    bool pushed = false;
    const bool up_first = (value - floor_value) > 0.5;
    for (int which = 0; which < 2; ++which) {
      const bool take_up = (which == 0) ? !up_first : up_first;
      Node child;
      child.lower = child_lower;
      child.upper = child_upper;
      child.depth = node.depth + 1;
      child.parent_bound = node_objective;
      child.branch_column = branch_column;
      child.branch_up = take_up;
      child.branch_fraction =
          take_up ? (ceil_value - value) : (value - floor_value);
      // The branched column is no longer free to take the cheaper rounding: in
      // this child it has taken the one the branch pinned. So the shared term
      // comes out and that side's own cost goes in.
      const double taken =
          take_up ? (ceil_value - value) *
                        (pseudo_up_n[sz(branch_column)] ? pseudo_up[sz(branch_column)] : 1.0)
                  : (value - floor_value) *
                        (pseudo_down_n[sz(branch_column)] ? pseudo_down[sz(branch_column)] : 1.0);
      child.estimate = estimate_base - branch_share + taken;
      if (options.warm_start &&
          options.node_solver == BranchAndBoundOptions::NodeSolver::kFirstOrder) {
        child.warm_x = relaxation.x;
      }
      child.basic = relaxation.basic;
      child.basis_status = relaxation.basis_status;

      if (take_up) {
        child.lower[sz(branch_column)] =
            std::fmax(child.lower[sz(branch_column)], ceil_value);
      } else {
        child.upper[sz(branch_column)] =
            std::fmin(child.upper[sz(branch_column)], floor_value);
      }
      if (child.lower[sz(branch_column)] > child.upper[sz(branch_column)]) continue;

      // The branch has just pinned this column to one side; propagation is what
      // turns that into information about the others.
      if (options.node_propagation) {
        Int before = 0;
        for (Int j = 0; j < working.num_cols(); ++j) {
          if (std::isfinite(child.lower[sz(j)])) ++before;
          if (std::isfinite(child.upper[sz(j)])) ++before;
        }
        const std::vector<double> before_lo = child.lower;
        const std::vector<double> before_hi = child.upper;
        if (!propagate_bounds(working, is_integral, &child.lower, &child.upper,
                              options.node_propagation_rounds)) {
          report(before_lo, before_hi, "propagation declared the child infeasible", 0.0);
          // Infeasible, and proved so by interval arithmetic rather than by a
          // relaxation that had to be solved first.
          ++result.children_pruned_by_propagation;
          continue;
        }
        Int after = 0;
        for (Int j = 0; j < working.num_cols(); ++j) {
          if (std::isfinite(child.lower[sz(j)])) ++after;
          if (std::isfinite(child.upper[sz(j)])) ++after;
        }
        if (after > before) result.propagation_tightenings += after - before;
      }

      stack.push_back(std::move(child));
      pushed = true;
    }
    // A node that produced children continues the plunge, up to the limit. The
    // limit is what stops a single dive from becoming the whole search again -
    // without it this is depth first with extra bookkeeping.
    if (pushed) {
      ++plunge_depth;
      plunging = options.max_plunge_depth < 0 || plunge_depth < options.max_plunge_depth;
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
      // An open node is not an unfinished one. Every node still on the stack
      // carries a bound its parent proved, and if the smallest of those has
      // already reached the cutoff then nothing left in the tree can beat what
      // is in hand - the nodes would each be discarded on that bound the moment
      // they were popped, and popping them would establish nothing that is not
      // established here.
      //
      // So this is the same proof the loop would have produced, read off the
      // stack instead of walked. It costs one pass over the open nodes and it
      // turns a run that stopped on the clock with the tree already decided into
      // the proof it had.
      if (best_open >= cutoff()) {
        status = MilpStatus::kOptimal;
        result.dual_bound = result.objective;
        result.relative_gap = 0.0;
      } else {
        result.dual_bound = to_model_objective(best_open);
        result.relative_gap = std::fabs(result.objective - result.dual_bound) /
                              std::fmax(1e-10, std::fabs(result.objective));
      }
    }
  } else {
    // No incumbent, whatever the reason. There is no solution to report, and
    // the field has to say so - it used to be left at its default zero on every
    // path except the infeasible one, so a run that found nothing came back
    // looking like a run that found an objective of zero.
    //
    // That is not cosmetic. On the wider MIPLIB set at a 15 second limit, seven
    // of the first thirteen instances end with no incumbent, and the survey
    // scored every one of them against the published optimum: acc-tight4, whose
    // optimum is 0, came back reading 0.000% error - an exact answer, from a
    // solve that never found a feasible point. 10teams read 100% error, which
    // is at least visibly wrong. Both are this missing line.
    //
    // A non-finite objective reaches the CLI as JSON null, which is what a
    // harness can actually test.
    result.objective = maximizing ? -kInf : kInf;
    // The bound is real even when the solution is not: every open node carries
    // a bound its parent proved, and the smallest of them bounds the whole
    // remaining tree. Worth reporting, since on an instance that finds nothing
    // it is the only thing the run established.
    double best_open = kInf;
    for (const Node& n : stack) best_open = std::fmin(best_open, n.parent_bound);
    result.dual_bound = std::isfinite(best_open) ? to_model_objective(best_open)
                                                 : (maximizing ? kInf : -kInf);
    // A gap needs two numbers and there is only one. Reporting zero here would
    // read as a closed gap on a solve that proved nothing.
    result.relative_gap = kInf;

    // "Infeasible" is a claim, and it is only true when every node was pruned
    // with a proof. If any relaxation ran out of budget without converging,
    // what we actually know is that we failed to solve the problem, and saying
    // infeasible instead would be reporting a wrong answer confidently. The
    // refinery MILP hit exactly this: its root relaxation does not converge, and
    // an earlier version of this code called the instance infeasible while
    // HiGHS solved it to optimality.
    if (status == MilpStatus::kInfeasible && result.nodes_relaxation_failed > 0) {
      status = MilpStatus::kRelaxationFailed;
      result.message =
          std::to_string(result.nodes_relaxation_failed) +
          " relaxation(s) did not converge, so infeasibility was never proved";
    }
  }

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
