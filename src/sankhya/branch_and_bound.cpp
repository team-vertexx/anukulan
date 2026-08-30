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
  // The parent's relaxation solution, used as a starting guess.
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

  auto solve_node = [&](const Node& node, NodeSolution* out,
                        Int iteration_factor = -1) {
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
      so.time_limit_seconds =
          std::max(0.0, options.time_limit_seconds - elapsed());
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
      SimplexResult r = solve_lp(lp, so);
      // Same rule as everywhere else this appears: a starting basis may save
      // pivots and may do nothing, but it may not cost an answer.
      if (root_cross.ok && so.start_basic == &root_cross.basic &&
          r.status != SimplexStatus::kOptimal) {
        SimplexOptions cold = so;
        cold.start_basic = nullptr;
        cold.start_status = nullptr;
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
      }
      return out->optimal;
    }

    const PdhgResult p = solve_pdhg(lp, options.relaxation);
    out->optimal = p.status == PdhgStatus::kOptimal;
    out->proved_infeasible = p.status == PdhgStatus::kPrimalInfeasible;
    out->x = p.x;
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
      if (elapsed() > options.root_cut_time_share * options.time_limit_seconds) break;
      // Through the same solver the nodes use, because that is what makes the
      // tableau available - and Gomory cuts are read off the tableau.
      Node probe;
      probe.lower = working.lower;
      probe.upper = working.upper;
      probe.basic = carry_basic;
      probe.basis_status = carry_status;
      NodeSolution root_solve;
      if (!solve_node(probe, &root_solve)) break;
      carry_from(root_solve);
      if (round == 0) {
        result.root_bound_before_cuts =
            sf.lp.objective_scale * sf.lp.standard_objective(root_solve.x) +
            sf.lp.objective_offset;
      }
      result.root_bound_after_cuts =
          sf.lp.objective_scale * sf.lp.standard_objective(root_solve.x) +
          sf.lp.objective_offset;
      root_std_after = working.standard_objective(root_solve.x);
      if (!have_root_std) {
        root_std_before = root_std_after;
        have_root_std = true;
      }

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
      working = append_cuts(working, cuts);
      extend_carry(rows_before_append);
      Node recheck;
      recheck.lower = working.lower;
      recheck.upper = working.upper;
      recheck.basic = carry_basic;
      recheck.basis_status = carry_status;
      NodeSolution after;
      if (solve_node(recheck, &after)) {
        carry_from(after);
        const double raised = working.standard_objective(after.x);
        if (raised < working.standard_objective(root_solve.x) - 1e-6) {
          working = before;
          result.cuts_rolled_back = true;
          break;
        }
      }
      for (const Cut& c : cuts)
        cut_family.push_back(std::string(c.family) + " round " +
                             std::to_string(round + 1));
      result.cuts_added += static_cast<Int>(cuts.size());
    }

    // Did any of that move the bound? The final root solve is the one that
    // answers it, since the loop above measures before adding the last round's
    // cuts.
    if (options.adaptive_cuts && have_root_std && result.cuts_added > 0) {
      Node final_probe;
      final_probe.lower = working.lower;
      final_probe.upper = working.upper;
      NodeSolution final_solve;
      if (solve_node(final_probe, &final_solve)) {
        root_std_after = working.standard_objective(final_solve.x);
        result.root_bound_after_cuts =
            sf.lp.objective_scale * root_std_after + sf.lp.objective_offset;
      }
      result.root_bound_rise = (root_std_after - root_std_before) /
                               std::fmax(1.0, std::fabs(root_std_before));
      if (result.root_bound_rise < options.cut_bound_improvement) {
        // The cuts cost every node below here and bought nothing measurable.
        working = uncut;
        result.cuts_discarded = true;
      }
    }
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

  MilpStatus status = MilpStatus::kInfeasible;
  double open_bound = -kInf;
  double pump_time_spent = 0.0;

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
    if (node.parent_bound >= incumbent) {
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

    if (node_objective >= incumbent) {
      report(node.lower, node.upper, "pruned on its own relaxation bound",
             node_objective);
      continue;  // bound prune
    }
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
              delta[side] = std::fmax(
                  0.0, working.standard_objective(probe_solve.x) - node_objective);
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
      const double gap = incumbent - node_objective;
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

    // Depth first, and the child nearer the relaxation value is explored first,
    // because it is the likelier place to find a feasible point early.
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
      if (options.warm_start) child.warm_x = relaxation.x;
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
