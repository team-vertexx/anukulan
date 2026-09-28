#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "anukulan/cholesky.hpp"
#include "anukulan/crossover.hpp"
#include "anukulan/pdhg.hpp"
#include "anukulan/scaling.hpp"
#include "anukulan/simplex.hpp"
#include "anukulan/standard_form.hpp"

namespace anukulan {

// Full definition only needed by callers that build one (presolve.hpp); a
// pointer is all solve_lp_race needs, so the header stays out of here.
class PostsolveStack;

// A primal-dual interior point method for linear programming: Mehrotra's
// predictor-corrector with Gondzio's multiple centrality correctors, on the
// normal equations, with the sparse Cholesky in cholesky.hpp.
//
// Where it sits among the engines here. The simplex walks vertices and its
// cost is the number of pivots, which on a large degenerate model can run to
// hundreds of thousands. The first-order method needs no factorisation at all
// and converges quickly to a rough answer and slowly to a sharp one. This
// method sits between them: every iteration is one factorisation, which is the
// expensive part, and the number of iterations is small and hardly depends on
// the size of the model - thirty to sixty on Netlib - because each one is a
// Newton step on the optimality conditions. What it does not produce is a
// vertex. Its answer lies in the middle of the optimal face, so it is handed to
// crossover_basis() and the simplex to finish exactly, like the first-order
// method's.
//
// The problem it iterates on is built from StandardLp (the first
// num_equalities rows are K x = q, the rest K x >= q, and lower <= x <= upper)
// in four moves:
//
//   - every inequality row gets a slack, K_i x - w_i = q_i with w_i >= 0, and
//     a two-sided model row - which the standard form splits into two rows,
//     one the negation of the other - is put back together as one row whose
//     slack is boxed, 0 <= w <= hi - lo. Two rows that are exact negatives
//     make the normal equations nearly singular for no reason;
//   - a fixed column is substituted out;
//   - every other column is shifted and, if it has only an upper bound,
//     reflected, so that it has a lower bound of zero, a finite upper bound
//     ("boxed"), or no bounds at all ("free");
//   - the whole thing is equilibrated first, by the same Ruiz scaling the
//     first-order method uses.
//
// Free variables are not split into a difference of two nonnegative ones,
// which makes the pair drift apart without bound and the normal equations
// singular in the limit. They are held by primal regularisation instead: the
// Newton matrix gets a small rho on their diagonal (Friedlander and Orban, "A
// primal-dual regularized interior-point method for convex quadratic
// programs", Math. Prog. Comp. 4, 2012), and the right-hand side is left
// alone, so the fixed point is still the unregularised optimum. A matching
// dual regularisation delta sits on the diagonal of the normal equations.
//
// A column with far more entries than the rest would make the normal
// equations dense - fit2p has 25 columns that between them touch all 3,000
// rows. Those are taken out of the factorisation and put back through the
// Sherman-Morrison-Woodbury identity, which is then only used as the
// preconditioner of conjugate gradients on the true system, because on its own
// it loses every digit on a row that the dense columns alone hold up
// (Andersen, Gondzio, Meszaros and Xu, "Implementation of interior point
// methods for large scale linear programming", 1996, section 5.2).
struct IpmOptions {
  // Relative primal residual, relative dual residual and relative duality
  // gap, all measured on the caller's unscaled problem with the same
  // definitions the first-order method uses (evaluate_residual in pdhg.hpp),
  // in both the 2-norm and the infinity norm. So a tolerance means the same
  // thing whichever of the two engines is asked.
  double tolerance = 1e-8;

  Int max_iterations = 200;
  double time_limit_seconds = 300.0;

  // Gondzio's multiple centrality correctors: after the Mehrotra direction,
  // up to this many further solves with the same factorisation, each trying
  // to push the complementarity products of the trial point back into a band
  // around the target so that a longer step fits. Gondzio, "Multiple
  // centrality corrections in a primal-dual method for linear programming",
  // Comput. Optim. Appl. 6, 1996.
  Int max_correctors = 2;

  // Fraction of the way to the boundary a step may go.
  double step_fraction = 0.995;

  // rho on the diagonal of every bounded variable, and on a free one's. The
  // free one's is all it has, so it decides how far a free variable may move
  // in one step; the bounded ones only cap Theta at 1/rho.
  double primal_regularization = 1e-10;
  double free_regularization = 1e-8;
  // delta on the diagonal of the normal equations.
  double dual_regularization = 1e-10;

  ScalingOptions scaling = ScalingOptions{10, false};
  bool scale = true;

  // A column is dense when it has more than max(dense_column_minimum,
  // dense_column_fraction * rows) entries, the model has at least
  // dense_column_rows rows, and there are no more than max_dense_columns of
  // them (the densest are taken). Each one costs a solve per iteration to put
  // back, so this is for a handful of columns, not for a dense model.
  double dense_column_fraction = 0.1;
  Int dense_column_minimum = 40;
  Int dense_column_rows = 200;
  Int max_dense_columns = 100;

  // Stop with a certificate when the iterates show the problem is primal or
  // dual infeasible. Only a certificate that checks out against the caller's
  // problem is reported; otherwise the method runs on to its limits and says
  // it did not converge.
  bool detect_infeasibility = true;

  CholeskyOptions cholesky;

  bool verbose = false;

  // Checked once per iteration and between the solves inside one. When it
  // reads true the method stops and reports kInterrupted with the best point
  // so far. This is how a race between engines stops the losers.
  const std::atomic<bool>* stop = nullptr;
};

enum class IpmStatus {
  kOptimal,
  kPrimalInfeasible,
  kDualInfeasible,
  kIterationLimit,
  kTimeLimit,
  kInterrupted,
  kNumericalError,
};

std::string to_string(IpmStatus status);

struct IpmResult {
  IpmStatus status = IpmStatus::kNumericalError;

  // In the caller's StandardLp, unscaled. x satisfies its bounds exactly; y is
  // non-negative on the inequality rows, as the standard form's dual has to be.
  // When the method did not converge these are the best point it reached,
  // measured by the largest of the relative residuals.
  std::vector<double> x;
  std::vector<double> y;
  double objective = 0.0;  // in the model's own sense, offset included

  Int iterations = 0;
  PdhgResidual residual;

  // The linear algebra, so a run can be read for where the time went.
  Int normal_rows = 0;
  long long normal_nonzeros = 0;   // lower triangle of the sparse part
  long long factor_nonzeros = 0;
  double factor_flops = 0.0;
  Int supernodes = 0;
  Int dense_columns = 0;
  Int ranges_merged = 0;
  Int fixed_columns = 0;
  Int free_columns = 0;
  Int dropped_pivots = 0;       // over the whole solve
  Int cg_iterations = 0;        // over the whole solve
  Int correctors = 0;           // centrality correctors accepted
  Int regularisation_raises = 0;
  double analyse_seconds = 0.0;
  double factor_seconds = 0.0;
  double solve_seconds = 0.0;

  // The last relative complementarity, mu over (1 + |objective|).
  double final_mu = 0.0;
  std::string message;
};

IpmResult solve_ipm(const StandardLp& lp, const IpmOptions& options = {});

// The race between the three engines this codebase now has for a general LP:
// the dual simplex cold, the interior point method finished by crossover and
// the simplex, and the first-order method finished the same way. `anukulan lp`
// runs the three on their own std::thread, each against the shared clock, and
// takes the first that reports an optimum which also survives the same check
// every simplex answer gets: postsolve applied and the row and bound violation
// measured against the model the caller actually handed in, not the one that
// was solved. Whichever wins sets the shared cancellation flag, which every
// engine below checks wherever it already checks its own time limit, so the
// other two unwind at their next chance rather than run to the clock.
//
// `lp` is the standard form actually solved, which is `original_model`'s own
// when the caller ran no presolve, and the presolve's reduced model's when it
// did - in which case `postsolve` recovers the row and column space the check
// runs in. Passing a null `postsolve` is how a caller with no presolve says
// its `lp` and `original_model` already agree.
struct RaceOptions {
  double time_limit_seconds = 60.0;

  bool use_simplex = true;
  bool use_ipm = true;
  bool use_pdhg = true;

  // Applies to every engine's own solve and to the simplex each of the three
  // finishes with. `algorithm`, `start_basic` and `start_status` are overwritten
  // per engine (the plain simplex engine runs dual, per solve_dual_simplex; the
  // other two run whatever this already says, primal by SimplexOptions' own
  // default, seeded from their crossover basis with a cold fallback exactly
  // like the simplex command's own --crossover path); everything else, the
  // tolerances, the pricing rule, the refactorisation frequency, is this
  // struct's as given.
  SimplexOptions simplex_options;
  IpmOptions ipm_options;
  PdhgOptions pdhg_options;
  CrossoverOptions crossover_options;

  // The first-order engine's own seed is capped the same way the simplex
  // command's --crossover path caps it: a seed run to full accuracy is the
  // first-order method's slow regime, and every iteration of it is an
  // iteration the simplex finish did not need. See command_simplex in
  // app/anukulan_cli.cpp for the sweep this was measured on.
  double pdhg_seed_tolerance = 1e-4;
  Int pdhg_seed_max_iterations = 5000;
  double pdhg_seed_iterations_per_row = 10.0;

  bool verbose = false;
};

// One engine's run, for a report of who did what.
struct RaceEngineOutcome {
  std::string name;  // "simplex", "ipm", "pdhg"
  bool ran = false;
  bool won = false;
  SimplexStatus status = SimplexStatus::kNumericalError;
  double seconds = 0.0;
  // Iterations of the engine's own method before it was handed to crossover;
  // 0 for the plain simplex engine, which has none.
  Int seed_iterations = 0;
  Int simplex_iterations = 0;
  std::string message;
};

struct RaceResult {
  // Empty when no engine produced an optimum that passed the check - every
  // engine ran out of clock, or the problem is infeasible or unbounded and
  // none of them was asked to prove that. The individual outcomes below still
  // say what each engine concluded.
  std::string winner;
  SimplexStatus status = SimplexStatus::kNumericalError;

  std::vector<double> x;  // original_model's own columns, postsolve applied
  std::vector<double> y;  // the winner's row duals, in the solved lp's rows
  double objective = 0.0;
  double row_violation = 0.0;
  double bound_violation = 0.0;

  std::vector<RaceEngineOutcome> engines;
  std::string message;
};

RaceResult solve_lp_race(const Model& original_model, const StandardLp& lp,
                         const PostsolveStack* postsolve, const RaceOptions& options = {});

}  // namespace anukulan
