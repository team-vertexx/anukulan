#pragma once

#include <string>
#include <vector>

#include "sankhya/lu.hpp"
#include "sankhya/standard_form.hpp"

namespace sankhya {

// The form the simplex actually computes in.
//
// Every row becomes an equality by giving each inequality its own slack, so the
// constraint matrix is [K | -I] over the inequality block and every constraint
// reads a'z = q. After that there are no row types left anywhere in the
// algorithm: a variable is either basic or sitting on one of its bounds, and
// that is the only case analysis the iteration needs.
struct LogicalForm {
  SparseMatrix columns;  // column-wise: row j holds column j of [K | -I]
  SparseMatrix rows;     // the same matrix row-wise
  std::vector<double> cost;
  std::vector<double> rhs;
  std::vector<double> lower;
  std::vector<double> upper;
  Int num_structural = 0;
  Int num_rows = 0;

  Int num_columns() const { return num_structural + num_rows; }
  Int num_equalities = 0;

  double objective_scale = 1.0;
  double objective_offset = 0.0;
};

LogicalForm to_logical_form(const StandardLp& lp);

// Where a non-basic variable is sitting. A basic variable's value comes from
// solving with the basis; everything else is pinned to a bound.
enum class VarStatus { kBasic, kAtLower, kAtUpper, kFree };

// The basis, its factorisation, and the quantities read off it.
class SimplexBasis {
 public:
  // Starts from the all-logical basis, which is the identity on the inequality
  // rows. Equality rows have no slack, so their basic variable has to come from
  // somewhere: the first structural column with an entry there is taken, which
  // is crude but gives a starting point that factorises.
  bool set_initial(const LogicalForm& form, std::string* error = nullptr);

  bool refactorize(const LogicalForm& form, std::string* error = nullptr);

  // B^-1 x and B^-T x for the basis as it stands, which is the factorisation
  // plus whatever pivots have happened since it was taken. Everything that
  // needs to solve with the basis goes through these rather than reaching for
  // the factors directly, because the factors alone are out of date the moment
  // a pivot lands.
  void ftran(std::vector<double>* x) const;
  void btran(std::vector<double>* x) const;

  // rho = B^-T e_row. The pivot row of the tableau is rho' N, which is what a
  // pricing rule needs to keep an edge-norm estimate up to date. One BTRAN.
  void pivot_row(const LogicalForm& form, Int row, std::vector<double>* rho) const;

  // x_B = B^-1 (q - N x_N), written into `values` for the basic positions and
  // the bound value for everything else.
  void compute_primal(const LogicalForm& form, std::vector<double>* values) const;

  // y = B^-T c_B, and the reduced costs d_j = c_j - y' a_j.
  void compute_duals(const LogicalForm& form, std::vector<double>* duals,
                     std::vector<double>* reduced_costs) const;

  const std::vector<Int>& basic() const { return basic_; }
  const std::vector<VarStatus>& status() const { return status_; }
  std::vector<VarStatus>& status() { return status_; }

  // Undoes the most recent update, putting the basis back to what it was one
  // pivot ago. The recovery path when a refactorisation fails: a stale product
  // form can report a pivotal column accurate enough to pass every test and
  // still pick a row whose true pivot is zero, and the basis that results is
  // genuinely singular rather than merely ill-conditioned. Refactorising it
  // cannot help; going back one pivot and refactorising that can.
  bool rollback_last_update();

  // Swaps `entering` in at row `leaving_row`, whose current occupant leaves to
  // the bound given.
  //
  // The new basis differs from the old one in a single column, so its inverse
  // differs by a single elementary factor and there is no need to build the
  // whole thing again. The factor is written from the pivotal column, which the
  // ratio test has already computed - hence the overload that takes it, and
  // hence the other one, which is the same thing for a caller that has not.
  //
  // Returns false when the pivot element is too small to divide by, which is
  // the caller's cue to refactorise and try again. Nothing is modified in that
  // case.
  bool pivot(const LogicalForm& form, Int leaving_row, Int entering,
             VarStatus leaving_to, const std::vector<double>& pivotal_column,
             std::string* error = nullptr);
  bool pivot(const LogicalForm& form, Int leaving_row, Int entering,
             VarStatus leaving_to, std::string* error = nullptr);

  // B a_q, the pivotal column.
  void ftran_column(const LogicalForm& form, Int column,
                    std::vector<double>* out) const;

  Int updates_since_refactorization() const { return updates_; }
  const LuFactor& factors() const { return lu_; }

  // The largest 1/|pivot| used since the last refactorisation. Each update
  // divides by its pivot element, so this is how much the representation can
  // have magnified a rounding error, and it is a better reason to refactorise
  // than a fixed pivot count. Small pivots are what makes the product form
  // decay; counting iterations only guesses at when that happened.
  double update_growth() const { return growth_; }

 private:
  // One elementary factor. B_new = B_old E, where E is the identity with its
  // `row` column replaced by the pivotal column, so B_new^-1 = E^-1 B_old^-1
  // and E^-1 is elementary too. Only the off-diagonal nonzeros are kept; the
  // diagonal is `pivot`.
  struct Update {
    Int row = 0;
    std::vector<Int> rows;
    std::vector<double> values;
    double pivot = 1.0;
    // Enough to undo it: who came in, who left, and where each of them was.
    Int entered = -1;
    Int left = -1;
    VarStatus entered_was = VarStatus::kAtLower;
  };

  std::vector<Int> basic_;
  std::vector<VarStatus> status_;
  LuFactor lu_;
  std::vector<Update> updates_list_;
  Int updates_ = 0;
  double growth_ = 1.0;
};

struct SimplexOptions {
  double primal_tolerance = 1e-7;
  double dual_tolerance = 1e-7;
  // A pivot smaller than this is refused: dividing by it would wreck the
  // factorisation regardless of what the ratio test wants.
  double pivot_tolerance = 1e-7;

  Int max_iterations = 200000;
  double time_limit_seconds = 300.0;
  // Measured, and measured twice, because the first answer stopped being true.
  //
  // With a linear phase one ratio test the product form stayed trustworthy to
  // 20 pivots between factorisations and broke at 25, where brandy reported a
  // feasible model infeasible, so this sat at 15 to keep a margin below the
  // cliff. The piecewise phase one removed the cliff rather than moving it:
  // every frequency from 15 to 200 now solves all sixteen instances, and the
  // whole set takes 2.71s at 15, 1.21s at 50 and 2.20s at 200.
  //
  // So a number chosen to survive a failure that no longer happens was costing
  // more than twice the time. 50 is the bottom of that curve.
  Int refactorization_frequency = 50;

  // Refactorising is driven by a count. A growth-based trigger was tried first
  // - refactorise once the largest 1/|pivot| since the last factorisation
  // passes a threshold - on the theory that a small pivot is what decays the
  // representation. It does not work: on brandy every threshold from 1e2 to
  // 1e12 fails the same way, and the worst growth sits at 4.4e6 regardless. The
  // measurement is still reported, because it is informative; it is just not a
  // criterion. What does work is rolling the updates back, below.

  // A degenerate pivot moves nothing: the step is zero, the basis changes, and
  // the objective does not. A run of them is how a simplex cycles, and Netlib's
  // brandy does exactly that - the same variable entering at the same row with
  // a zero step, forever.
  //
  // After this many consecutive zero steps the pricing and the ratio test both
  // switch to Bland's rule, which takes the lowest eligible index at both ends
  // and cannot cycle. It is a slow rule, which is why it is not the default;
  // the moment a real step is taken the solver goes back to Dantzig. Bland,
  // "New finite pivoting rules for the simplex method", Math. of OR 2(2), 1977.
  Int stall_iterations = 20;

  // How far back a refactorisation may walk before giving up.
  Int max_rollback = 20;

  // A step below this counts as degenerate for the purpose above.
  double degenerate_step = 1e-9;

  // Walk phase one's breakpoints and stop where the slope of the sum of
  // infeasibilities turns non-negative, instead of at the first blocking row.
  // Phase one only - in phase two the objective really is linear and the
  // ordinary test is right.
  //
  // Measured over sixteen Netlib instances: 16/16 either way, 32% fewer phase
  // one iterations geomean, and the whole set in 1.46s against 1.60s. The
  // averages undersell it, because what it really did was remove the reasons
  // for three other things:
  //
  //   brandy goes from 20,109 iterations to 839, and from 0.61s to 0.03s. It
  //     stops needing Bland's rule at all - 26 switches to none - so its
  //     cycling was a symptom of stopping at the first breakpoint, not
  //     something intrinsic to the model;
  //   brandy also stops caring about the refactorisation frequency, solving at
  //     15, 25, 50 and 100 where before it produced a false infeasibility past
  //     20;
  //   fit1p under Devex goes from "infeasible" to the published optimum. That
  //     was the defect this was written for: a variable heading back into its
  //     bounds with an infinite opposite bound blocked on nothing, and the step
  //     came out unbounded. Here the slope turns at the violated bound whether
  //     or not the far one exists, so the case cannot arise.
  //
  // fit1p under Dantzig is the one that got worse, 3260 iterations to 6747.
  bool piecewise_phase_one = true;

  bool verbose = false;
  Int log_frequency = 1000;

  // How the entering column is chosen.
  //
  // Dantzig takes the largest reduced cost, which is scale dependent: a column
  // whose d_j is large because its units are small is not a column that moves
  // the objective far. Steepest edge divides d_j by the norm of the edge it
  // would move along, which is exact and costs a solve per column. Devex
  // (Harris, Math. Prog. 5, 1973) approximates that norm with a weight carried
  // forward and corrected at each pivot, for one extra BTRAN and one extra pass
  // over the columns per iteration.
  //
  // Devex is the default, and that reverses an earlier decision made on an
  // earlier measurement. Against a linear phase one ratio test it took 1.13x
  // fewer iterations for 1.01x the wall clock - the extra BTRAN and column pass
  // ate the saving exactly - and it failed fit1p outright, so Dantzig kept the
  // default. With the piecewise phase one it takes 0.79x the iterations and
  // solves all sixteen instances in 0.76s against Dantzig's 1.25s.
  //
  // Neither number was wrong when it was taken. The ground moved: most of what
  // Devex was buying was being spent again in a phase one that stopped at the
  // first breakpoint, and its one failure was that phase one's hole rather than
  // anything about pricing.
  //
  // What pricing can and cannot do is worth keeping straight. It only decides
  // which improving column to take; every column it may pick still has a
  // favourable reduced cost, and the ratio test is untouched. A wrong weight
  // makes the solver slower, never wrong.
  enum class Pricing { kDantzig, kDevex };
  Pricing pricing = Pricing::kDevex;

  // Primal or dual. The primal walks feasible points until no column improves;
  // the dual keeps the reduced costs feasible and drives the basic variables
  // into their bounds.
  //
  // The dual is here for branch and bound. Tightening one bound on one variable
  // leaves the parent's basis dual feasible and only makes it primal infeasible
  // in that one row, so a child node re-optimises in a handful of pivots
  // instead of solving from scratch. That is the whole reason node relaxations
  // are affordable, and it is why every serious MILP code solves nodes with a
  // dual simplex.
  enum class Algorithm { kPrimal, kDual };
  Algorithm algorithm = Algorithm::kPrimal;

  // The weights drift away from the true edge norms as the reference framework
  // ages. Past this the framework is reset to the current nonbasic set and
  // every weight goes back to one.
  double devex_reset_weight = 1e6;
};

enum class SimplexStatus {
  kOptimal,
  kUnbounded,
  kInfeasible,
  kIterationLimit,
  kTimeLimit,
  kNumericalError,
};

std::string to_string(SimplexStatus status);

struct SimplexResult {
  SimplexStatus status = SimplexStatus::kNumericalError;
  std::vector<double> x;  // structural variables only, in model order
  std::vector<double> y;  // row duals
  double objective = 0.0;

  Int iterations = 0;
  Int phase_one_iterations = 0;
  Int refactorizations = 0;
  // How many times pricing switched into or out of Bland's rule. A nonzero
  // count means the model stalled and the anti-cycling rule earned its keep.
  Int bland_switches = 0;
  // The largest 1/|pivot| the product form was asked to divide by, over the
  // whole solve. Says how close the updates came to being unusable.
  double worst_update_growth = 1.0;
  // Pivots undone because the basis they produced would not factorise.
  Int rollbacks = 0;
  // Times a conclusion was held back until the basis had been refactorised.
  Int confirmations = 0;
  // Times the Devex reference framework was restarted.
  Int devex_resets = 0;
  // Nonbasic columns moved to their other bound to make the start dual
  // feasible, and whether the dual had to hand back to the primal.
  Int dual_start_flips = 0;
  bool fell_back_to_primal = false;
  double solve_seconds = 0.0;
  std::string message;
};

// Primal simplex with bounded variables. Phase one drives the basis to
// feasibility by minimising the total bound violation; phase two then optimises.
SimplexResult solve_simplex(const StandardLp& lp, const SimplexOptions& options = {});

// Dual simplex, following Koberstein's Algorithm 2 (The Dual Simplex Method,
// 2005). Requires a dual feasible start, which for a boxed variable is a matter
// of putting it on the bound whose sign matches its reduced cost - so the
// starting basis is made dual feasible by flipping bounds, not by a phase one.
//
// A variable whose reduced cost has the wrong sign and no finite bound on that
// side cannot be flipped, and there is no dual phase one here. When that
// happens the result says so and the caller should use the primal; solve_lp
// below does that automatically.
SimplexResult solve_dual_simplex(const StandardLp& lp,
                                 const SimplexOptions& options = {});

// Picks by options.algorithm, and falls back from dual to primal when the start
// cannot be made dual feasible.
SimplexResult solve_lp(const StandardLp& lp, const SimplexOptions& options = {});

}  // namespace sankhya
