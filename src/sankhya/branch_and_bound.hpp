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

  // Time is the resource that matters; the node limit is a backstop against a
  // tree that is going nowhere, and it should not be what stops a solve that
  // was about to finish.
  //
  // It used to be 20,000, and that was doing the wrong job. flugpl proves
  // optimality at 28,917 nodes in half a second - comfortably inside the time
  // limit and well past the node one - so the solver was reporting a 3.2% error
  // on an instance it could finish, and stopping to do it. Raising the limit
  // takes the seven-instance set from three proved optimal to four, and on the
  // three that still do not finish the *solution* is already optimal or within
  // 0.14%: what is missing is the proof, not the answer.
  Int node_limit = 1000000;
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
  // Share of the whole time budget the root cut loop may spend. Eight rounds
  // are eight full LPs on a relaxation that is growing by up to sixty rows a
  // round, and on a hard instance that is the entire run: berlin at a fifteen
  // second limit solved eleven relaxations for 75,098 simplex iterations and
  // then explored *zero* nodes, so it reported no incumbent and no bound
  // improvement, having spent the budget tightening a root it never used.
  //
  // Cuts are worth a slice, not the run. Whatever rounds fit inside the slice
  // are kept; the rest are dropped and the tree gets the remaining time.
  double root_cut_time_share = 0.3;
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
  // A solution known to be feasible and optimal, for debugging. When set, every
  // point at which a node could be discarded checks whether that solution is
  // inside the node's box first, and reports the first place one that contains
  // it is thrown away.
  //
  // A branch-and-bound that returns a wrong answer has pruned the optimum, and
  // there is no way to reason backwards from the wrong answer to the prune that
  // caused it. This turns that question into a printed line. SCIP carries the
  // same facility for the same reason.
  const std::vector<double>* debug_solution = nullptr;

  bool reduced_cost_fixing = true;

  // Use the fact that every feasible objective value is a whole number, when it
  // is one, to strengthen the cutoff by a full unit. See the detection in
  // branch_and_bound.cpp for the argument. Switchable so its effect can be
  // measured rather than assumed - it is exact either way, so the only question
  // is whether the extra prunes are worth the search order they change.
  bool objective_integrality = true;

  // Propagate bounds into each child as it is created. Branching has just
  // pinned a variable to one side of its fractional value, which is precisely
  // the moment interval arithmetic on the rows has something new to work with -
  // it can tighten other variables, and sometimes prove the child infeasible
  // before its relaxation is ever solved.
  //
  // The machinery is the same `propagate_bounds` the fix-and-propagate
  // heuristic uses; this only points it at the children. Rounds are kept low
  // because this runs at every node and the first round finds most of it.
  // On, and the round count is measured. Six MIPLIB instances at a 45 second
  // budget, in nodes where the instance solves and in remaining gap where it
  // does not:
  //
  //                 off    1 round   2 rounds   4 rounds
  //   flugpl      28917      2283        979        477
  //   gt2          1181       826        789        783
  //   khb05250      143       143        142        142
  //   p0201      6.076%      2364       2212       2282
  //   gen-ip054  1.700%    1.700%     1.700%     1.700%
  //   mas76      4.192%    4.192%     4.192%     4.192%
  //
  // No instance is worse for it, and two are transformed: flugpl needs a
  // sixty-first of the tree, and p0201 stops failing to finish inside the
  // budget. More rounds keeps helping where it helps at all - flugpl halves
  // again from two rounds to four - and the only cost anywhere is p0201's
  // seventy extra nodes, against flugpl's five hundred saved.
  // Reliability branching, in the sense of Achterberg, Koch and Martin: strong
  // branch on a candidate until enough is known about it, then trust its
  // pseudocost from there on.
  //
  // Pure pseudocost branching has to guess at a variable it has never branched
  // on, and the guess is the same optimistic unit cost for all of them, so
  // early decisions near the root - the ones that shape the whole tree - are
  // made with no information. Full strong branching removes the guess and is
  // reported at around 65% fewer nodes for up to 44% more time, which is the
  // wrong trade. Reliability spends the strong branching only where the
  // pseudocost is not yet trustworthy, which is mostly near the root.
  //
  // A variable counts as reliable once both its directions have been measured
  // this many times.
  // Measured, six MIPLIB instances at a 45 second budget - nodes where the
  // instance solves, remaining gap where it does not. Threshold 2 throughout,
  // varying only how deep the strong branching is allowed to go:
  //
  //                off       d=3       d=6      d=10   no limit
  //   flugpl       477       246       246       246        246
  //   gt2          783      1509       374       194       2225
  //   khb05250     142        79        76        86         78
  //   p0201       2282       804       873       920        776
  //   gen-ip054  1.700%    1.714%    2.192%    2.192%     2.047%
  //   mas76      2.778%    3.565%    3.565%    4.099%     4.047%
  //
  // The depth cap is not a refinement here, it is the whole thing. Unlimited,
  // gt2 goes from 783 nodes to 2,225 - strong branching makes it nearly three
  // times worse. Capped at ten it goes to 194. Across the four that solve, the
  // total is 3,684 nodes against 1,446, so 0.39x.
  //
  // Why the cap matters that much: near the root a branching decision shapes
  // the whole tree and is worth paying to get right, and deep down it settles a
  // subtree that is about to be pruned anyway, where the probes are pure cost
  // and the greedy one-level-ahead choice is not the one that makes the
  // smallest tree.
  //
  // What it costs: gen-ip054 and mas76 both end with worse gaps, 1.700% to
  // 2.192% and 2.778% to 4.099%. Neither finishes either way, and the probes
  // spend time those two would otherwise put into nodes. Worth saying plainly
  // that on mas76 the *solution* is the published optimum in both cases - what
  // gets worse there is the proof, not the answer.
  bool reliability_branching = true;
  Int reliability_threshold = 2;
  // At most this many unreliable candidates get strong branched at one node,
  // best-first by their current score. Each costs two bounded LP solves.
  Int strong_branch_candidates = 8;
  // Iteration budget for one strong branch probe, as a multiple of the row
  // count. Small on purpose: the point is a usable estimate of the bound
  // change, not the child's exact value.
  Int strong_branch_iteration_factor = 5;
  // Only strong branch this far down. Near the root a branching decision shapes
  // the whole tree and is worth paying to get right; deep down it settles a
  // subtree that is about to be pruned anyway, and the probes are pure cost.
  // Negative means no limit.
  Int strong_branch_max_depth = 10;

  bool node_propagation = true;
  int node_propagation_rounds = 4;

  // Whether a cut family is worth its cost is a property of the instance, not
  // of the family. Gomory takes khb05250 from 4,247 nodes to 143 and makes four
  // other instances worse, and no single on-or-off answer expresses that.
  //
  // So: generate the cuts, then ask whether they moved the root bound, and keep
  // them only if they did. The measure is the relative rise in the standard-form
  // root objective, which needs nothing the solver does not already have - in
  // particular it does not need the optimum.
  //
  // Over the seven instances, that rise sorts them exactly:
  //
  //   gt2        55%     tree 1.01x     keep
  //   khb05250   10.2%        0.03x     keep
  //   p0201       4.7%        0.88x     keep
  //   neos5       1.3%        worse     drop
  //   flugpl      0.43%       1.31x     drop
  //   mas76       0.18%       worse     drop
  //   gen-ip054   0.015%      worse     drop
  //
  // The three that gain are the three largest rises and the four that lose are
  // the four smallest, with a factor of three between the classes. Seven
  // instances is a small sample to fit a threshold on and this could be
  // overfitted; what argues against that is the mechanism, which is not a
  // correlation - cuts that do not move the bound have made every node more
  // expensive and bought nothing.
  // Individually switchable so an invalid cut can be traced to its family.
  bool cover_cuts = true;
  bool mir_cuts = true;

  bool adaptive_cuts = true;
  double cut_bound_improvement = 0.02;

  // After the root cut loop, keep only the cuts the root optimum actually sits
  // on, and rebuild the relaxation without the rest.
  //
  // A constraint slack at an optimal point is not what is holding it there, so
  // removing it leaves that point feasible and optimal with the zero multiplier
  // it already had. The root bound is therefore unchanged by construction, not
  // by measurement, and every node below solves a smaller LP for it.
  //
  // The cost is real but deferred: branching moves the relaxation, and a cut
  // slack at the root can bind once a variable is pinned. A cut pool would keep
  // those and re-add them where they bite, which is what cuts.hpp says is
  // missing and still is. This is the part of cut management that fits a tree
  // whose nodes all share one matrix - keep what did the work at the root, drop
  // what only costs.
  bool root_cut_filtering = true;

  // Now on, because the rule above is what decides per instance.
  bool gomory_cuts = true;

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

  // LP-guided diving, in the sense of SCIP's fracdiving.
  //
  // The fix-and-propagate dive above reads every one of its decisions off a
  // single relaxation: it sorts the integer columns by how fractional they were
  // at this node, then pins each in turn to the nearest integer, propagating
  // between pins to catch a doomed rounding early. It never re-solves. So after
  // twenty pins it is still steering by a solution of a problem that had none
  // of them, and the further it goes the less that solution has to say.
  //
  // Re-solving after each decision fixes exactly that, and here it is close to
  // free. A dive step tightens one bound on one variable, which is precisely
  // what a branching child does, so the parent's basis is still dual feasible
  // and the dual simplex re-optimises in a handful of pivots - the same warm
  // start that makes the tree affordable at all. What the dive gets for it is a
  // relaxation that has already absorbed every decision so far.
  //
  // Bound rather than fix: a column at 2.3 gets an upper bound of 2, not the
  // value 2. The rounded side is a branching child and keeps the rest of the
  // range live, where fixing throws away everything the LP might still have
  // wanted. Where the rounded side comes back infeasible, the other side is
  // tried once, and a step with neither side available ends the dive.
  bool lp_diving = true;
  // The column whose value is nearest an integer goes first. It is the smallest
  // disturbance available, so it is the decision least likely to make the
  // relaxation infeasible, and rounding it is the decision the relaxation is
  // most nearly making already.
  Int lp_dive_max_steps = 200;
  // Iterations one dive re-solve may take, as a multiple of the row count. It
  // is a warm started re-optimisation after a single bound change, so it should
  // be quick; a long one means this is not the cheap thing it is supposed to be.
  Int lp_dive_iteration_factor = 20;
  // Share of the whole budget all dives together may spend. Same rule as the
  // pump: worth a lot when it works, nothing when it does not, so it gets a
  // slice rather than the run.
  double lp_dive_time_share = 0.2;
  // Retry every this many nodes while there is still no incumbent. A dive from
  // a different relaxation makes different decisions, so a second attempt is
  // not the first one repeated.
  Int lp_dive_retry_nodes = 100;

  // Feasibility pump, following Fischetti, Glover and Lodi (Math. Prog. 104,
  // 2005). Rounding and diving both work forwards from the relaxation: round it
  // and hope, or pin variables one at a time and hope the propagation survives.
  // On the wider MIPLIB set neither finds anything on most instances - a
  // fifteen second run of the first thirteen ended with no incumbent at all on
  // seven of them, so there was nothing to prune against and the tree was
  // exploring blind.
  //
  // The pump works the other way round. It keeps two points: an integral one
  // that need not satisfy the rows, and a relaxation point that satisfies the
  // rows but need not be integral. Each round rounds the second to refresh the
  // first, then re-solves the relaxation with the objective replaced by the
  // distance to it. The two are pulled together until they meet, and where they
  // meet is a feasible integer point.
  //
  // What makes it affordable: only the objective changes between rounds, so the
  // bounds and rows are untouched and the previous basis is still primal
  // feasible. The primal simplex warm starts from it and finishes in a few
  // pivots, exactly as a branch and bound child does for the dual.
  bool feasibility_pump = true;
  Int pump_max_rounds = 60;
  // Two identical roundings in a row means the distance LP has stopped moving.
  // The escape is to flip some of the least confident columns and carry on;
  // past this many escapes the pump is not going to find anything.
  Int pump_max_restarts = 10;
  // Iterations one distance LP may take, as a multiple of the row count. It is
  // a warm started re-optimisation after an objective change, so it should be
  // quick; a long one means the pump is not the cheap thing it is supposed to
  // be.
  Int pump_iteration_factor = 10;
  // Share of the pump's own time budget, as a fraction of the whole solve. The
  // pump is worth a lot when it works and nothing when it does not, so it gets
  // a slice rather than the run.
  double pump_time_share = 0.25;
  // Retry the pump every this many nodes while there is still no incumbent. A
  // pump started from a different relaxation rounds to a different point, so a
  // second attempt is not the first one repeated. Zero means root only.
  Int pump_retry_nodes = 200;
  // Objective feasibility pump, following Achterberg and Berthold (CPAIOR
  // 2007). The plain pump chases feasibility alone, so the first point it finds
  // is wherever the rounding happened to lead - and it cycles, because nothing
  // distinguishes the many roundings that are equally close. Mixing a decaying
  // share of the real objective into the distance both biases the search toward
  // solutions worth having and breaks the symmetry that causes the cycling.
  //
  //   objective = (1 - a) * distance / ||distance|| + a * c / ||c||
  //
  // starting at this weight and multiplied by pump_objective_decay each round,
  // so the first rounds follow the objective and the later ones follow
  // feasibility. Zero is the plain pump.
  double pump_objective_weight = 0.0;
  double pump_objective_decay = 0.9;

  // Seed the root relaxation's basis from a short first-order solve, the way
  // crossover.hpp describes. A child inherits its parent's basis and needs a
  // handful of pivots; the root inherits nothing and pays for the whole walk,
  // and on the wider set that walk is what consumes the budget - 10teams and
  // binkar10_1 both spend fifteen seconds without finishing a root relaxation,
  // so no node is ever explored and no heuristic ever runs.
  //
  // This is also where the GPU earns its place in the MILP half. The seed is
  // matrix-vector products and clamps, which is the part that goes on a device;
  // the pivots it saves are the part that does not.
  bool root_crossover = true;
  double root_crossover_tolerance = 1e-4;
  Int root_crossover_max_iterations = 5000;
  // The same per-row budget the LP path uses; see the sweep in the CLI. A flat
  // cap is too loose for a small root and too tight for a large one, and the
  // roots that need this most are the large ones.
  double root_crossover_iterations_per_row = 10.0;
  // Share of the remaining budget the seed may use. An iteration cap bounds
  // work, not time, and on a large enough model those are different things.
  double root_crossover_time_share = 0.2;

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
  // Feasibility pump: rounds run, times it produced an incumbent, and how many
  // times it had to escape a cycle.
  Int pump_rounds = 0;
  Int pump_successes = 0;
  Int pump_restarts = 0;
  // Columns the root crossover pushed into the starting basis, and whether the
  // simplex accepted the basis it produced.
  Int root_crossover_pushed = 0;
  bool root_crossover_used = false;
  // Bounds tightened by reduced-cost fixing, and how many closed a range to nothing.
  Int reduced_cost_tightenings = 0;
  Int reduced_cost_fixings = 0;
  // Cuts generated and then thrown away because they did not move the bound.
  // Children proved infeasible by propagation, so never solved at all, and
  // bounds tightened on the ones that survived.
  Int children_pruned_by_propagation = 0;
  Int propagation_tightenings = 0;
  // Strong branch probes run, and how many of them proved a child infeasible
  // outright - which is a pruned subtree found for the price of a bounded solve.
  Int strong_branch_probes = 0;
  Int strong_branch_prunes = 0;
  bool cuts_discarded = false;
  // Cuts thrown away after the root because the root optimum did not sit on
  // them. Not the same as cuts_discarded, which throws away the whole effort.
  Int cuts_dropped_slack = 0;
  double root_bound_rise = 0.0;
  Int dives_run = 0;
  // LP-guided dives: how many were started, how many produced an incumbent, and
  // how many relaxations they solved between them. Solves over dives is the
  // number to look at - a dive that ends after two is being killed by
  // infeasibility, not by the budget.
  // Whether the objective was found to take only whole-number values, which is
  // what lets the cutoff move a unit below the incumbent.
  bool integral_objective = false;
  Int lp_dives_run = 0;
  Int lp_dive_successes = 0;
  Int lp_dive_steps = 0;
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
