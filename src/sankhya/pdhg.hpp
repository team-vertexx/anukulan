#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "sankhya/scaling.hpp"
#include "sankhya/standard_form.hpp"

namespace sankhya {

// Primal-dual hybrid gradient for linear programming, in the restarted-average
// form of PDLP and cuPDLP.
//
// The whole iteration is two sparse matrix-vector products and some vector
// arithmetic. There is no factorization and no basis, which is exactly why this
// is the algorithm that ports to a GPU.
struct PdhgOptions {
  // One relative tolerance, applied to the primal residual, the dual residual
  // and the duality gap. 1e-4 is the "moderate accuracy" setting first-order
  // methods are usually benchmarked at; 1e-8 is the high accuracy setting.
  double tolerance = 1e-6;

  // Tolerance on the duality gap alone, when it should differ from the one
  // above. Zero means "the same as `tolerance`", which is the usual thing to
  // want and what every caller got before this existed.
  //
  // It exists because of what feasibility polishing is for. Section 4 of the
  // PDLP paper is explicit that the technique targets "extremely small
  // feasibility violations and moderate (e.g. 1%) duality gaps" - it buys
  // feasibility, and it pays for it in the gap. Asking for 1e-9 feasibility and
  // a 1% gap is a perfectly sensible thing to want from a production model: a
  // refinery plan that violates a capacity is not a plan, while a plan that
  // leaves 1% of margin on the table is still a plan. Asking for both at 1e-9
  // is what simplex is for.
  double gap_tolerance = 0.0;

  Int max_iterations = 200000;
  double time_limit_seconds = 300.0;

  // Convergence is checked periodically rather than every iteration: the check
  // costs two more matrix-vector products, which is as much as an iteration.
  Int termination_check_frequency = 40;

  // Additionally require that no single constraint is violated by more than this
  // in absolute terms. Zero disables it, which is what PDLP does. A positive
  // value is the safer default on models whose right-hand side is large, where
  // the relative 2-norm criterion alone lets real violations through.
  double absolute_tolerance = 0.0;

  // Require the infinity-norm relative criteria as well as PDLP's 2-norm ones.
  // On by default: the 2-norm form on its own reported "optimal" on
  // irish-electricity at a point whose objective was 3.6% wrong. Turn it off to
  // reproduce plain PDLP behaviour.
  bool require_inf_norm_termination = true;

  // Each of these is a separate piece of machinery, kept switchable so their
  // effect can be measured one at a time rather than asserted.
  bool adaptive_step_size = true;
  bool primal_weight_updates = true;
  bool restarts = true;

  // Halpern iteration instead of the averaged restart scheme.
  //
  // Plain PDHG restarted on a running average is raPDHG - what this has done
  // until now. Restarted Halpern PDHG keeps an anchor instead: each step is a
  // combination of the PDHG operator's output and the point the current epoch
  // started from,
  //
  //     z^{k+1} = (k+1)/(k+2) T(z^k) + 1/(k+2) z^{0}
  //
  // and a restart takes one plain PDHG step and makes that the new anchor.
  // Lu and Yang, Restarted Halpern PDHG for Linear Programming
  // (arXiv:2407.16144); it is what cuPDLPx is built on, and the reported
  // speedups over averaged restarts are large.
  //
  // On, and here is what that rests on. Over the sixteen Netlib instances with
  // published optima, solved to 1e-6 with an absolute cap and presolve: 0.87x
  // the iterations and 0.86x the wall clock, correct on all sixteen either way.
  // Eleven of the sixteen improve, and the largest do most - 25fv47 140,520
  // iterations to 102,920, maros-r7 8.4s to 5.5s, stocfor2 2.1s to 1.3s.
  //
  // Where it does not help: qap15 and datt256_lp are neutral at every tolerance
  // from 1e-4 to 1e-8, and graph40-40 is about 1.4x worse. That last one is the
  // GPU demonstration instance, but the speedup there is CPU against GPU on the
  // same algorithm, so both sides move together and the ratio does not change.
  //
  // The literature reports far more than 1.16x for this - cuPDLPx claims 2.5x
  // to 5x on MIPLIB relaxations - and the gap is not a mystery: that solver
  // also has a new restart criterion and a PID-controlled primal weight update,
  // neither of which is here. This is the Halpern base on its own.
  bool halpern = true;

  // Restart on the Halpern paper's own test - equation (10) of Lu and Yang,
  // where an epoch ends once ||z - T(z)|| has decayed by a factor of e - rather
  // than on the KKT error the averaged scheme uses.
  //
  // It is off, and that is measured. Over the sixteen Netlib instances Halpern
  // with the existing KKT restart takes 0.87x the iterations and 0.86x the time
  // of averaged restarts; with the paper's test instead it is 0.93x and 0.91x.
  // The KKT restart is PDLP's and has had a lot of tuning behind it, and it
  // keeps winning here. The paper's test is cheaper - it reuses a quantity the
  // adaptive step size already computes - so it may yet be the better one with
  // a floor chosen by measurement rather than by guess.
  bool halpern_restart = false;

  // Shortest epoch the Halpern restart test will end. Without a floor the test
  // can fire on the first iteration of an epoch and restart forever. Chosen by
  // guess, which is the honest reason not to trust the comparison above too
  // far.
  Int halpern_minimum_epoch = 16;

  // Feasibility polishing: Algorithm 4 of Applegate, Diaz, Hinder, Lu, Lubin,
  // O'Donoghue and Schudy, "PDLP: A Practical First-Order Method for
  // Large-Scale Linear Programming" (arXiv:2501.07018), section 4.
  //
  // Two insights, both theirs. A feasibility problem - the same constraints
  // with no objective - is empirically orders of magnitude faster for PDHG than
  // the LP it came from, because nothing is pulling against the constraints.
  // And PDHG's iterates are non-increasing in distance to any optimal solution,
  // so a run warm started near a good point stays near it. Put together: warm
  // start a feasibility solve from a near-optimal iterate and you land on a
  // feasible point with roughly the objective you started with.
  //
  // Two sub-problems, both solved by this same routine:
  //
  //   primal   c := 0                        started from (x, 0)
  //   dual     q := 0, finite bounds := 0    started from (0, y)
  //
  // The second is the paper's problem (7), "maximize 0 subject to c - A'y = r,
  // y in Y, r in R": zeroing the right-hand side and the bound values leaves
  // the dual feasible set alone - it reads K, c and which bounds are finite,
  // none of which move - while making the dual objective identically zero.
  //
  // Each sub-solve starts the *other* coordinate at zero rather than carrying
  // the seed's. Zero is trivially feasible for the sub-problem's other side, so
  // it contributes no force; carrying over a dual that was tuned for an
  // objective the sub-problem no longer has does contribute one, and the
  // iterate walks away. Measured here on 25fv47 with the seed's dual carried
  // in: feasibility polished to 1e-10 and 1e-9, and the dual objective went
  // from 5969 to -11234.
  //
  // On, and here is what that rests on. Eighteen Netlib instances with
  // published optima, presolved, against the same run with polishing off, and
  // counting the polishing sub-solves' iterations as the work they are:
  //
  //   one tolerance on everything (--tol=1e-8)   1.00x iterations, 1.00x time
  //   feasibility 1e-8, gap 1e-2                 0.79x iterations, 0.88x time
  //
  // The first line is the point of the first: with no slack in the gap the
  // trigger almost never fires, so leaving this on costs nothing when the
  // caller wants everything tight. The second is what the technique is for,
  // and the gains sit on the instances that need them - fit1p 0.06x, bandm
  // 0.11x, maros-r7 0.12x, 25fv47 0.12x, adlittle 0.51x, degen3 0.61x - while
  // the losses are all on instances that finish in under two thousand
  // iterations anyway (sctap1 1.20x, at a hundredth of a second). bandm stops
  // reporting a numerical error and starts reporting an answer.
  //
  // The objective errors get worse, and that is the trade, not a defect: 25fv47
  // goes from 1.5e-07 to 6.8e-03 because a 1% gap is what was asked for.
  // Anyone wanting both tight is asking for the simplex.
  //
  // On the refinery model, at the defaults this ships with: 160,720 iterations
  // and a capacity violated by 1.5e-02 without it, against 12,800 plus 1,720 of
  // polishing and a violation of 1.3e-04 with it. Eleven times fewer iterations
  // for a violation 114 times smaller, paid for with a 0.45% gap.
  //
  // It used to read better than that - before the cuPDLPx additions the
  // unpolished solve hit the iteration limit outright, so polishing was the
  // difference between an answer and none. The base method getting faster
  // turned that into the difference between an answer and a worse one, which is
  // the better problem to have and worth writing down rather than quietly
  // keeping the older, more flattering number.
  //
  // What is not here: a back-off after repeated failures. greenbea spends 1.11x
  // and gains nothing, which is the worst case in the set, but it only makes
  // one attempt before the iteration limit while 25fv47 needs three before one
  // is accepted - so any cutoff cheap enough to help greenbea also throws away
  // 25fv47's 0.12x.
  bool polish_feasibility = true;

  // The gap is what polishing spends, so the gap is what triggers it: the paper
  // pauses normal iterations once the relative duality gap is already under the
  // target, and asks polishing for the feasibility. No point paying for a
  // polish whose result the gap test will reject anyway.
  //
  // Attempts are spaced at doubling iteration counts - 100, 200, 400, 800 -
  // which is what bounds the total cost. Each attempt gets a budget
  // proportional to the iterations behind it, so a geometric schedule keeps the
  // sum proportional to the work already done rather than to the number of
  // checks.
  Int polish_first_iteration = 100;

  // Budget for one sub-solve, as a fraction of the iterations taken so far. The
  // paper's k/8.
  double polish_iteration_factor = 0.125;
  // A sub-solve only gets to stop early at a termination check, so a budget
  // below one check period is spent in full whatever happens. The floor is one
  // check period for that reason, not a tuned number.
  Int polish_minimum_iterations = 40;
  Int polish_maximum_iterations = 20000;

  // Polish once more after the main loop stops. The periodic polish is about
  // finishing sooner; this one is about the answer that gets handed back. A
  // first-order method that runs out of iterations returns a point with a real
  // constraint violation in it, and this is the chance to hand back a feasible
  // one instead. Accepted only when neither feasibility measure gets worse, at
  // least one gets better, and the gap stays inside what the caller already
  // said they would accept.
  //
  // It fires when the main loop ran out of iterations, and not when it ran out
  // of time - there is no time left to polish in, and quietly running past a
  // deadline is worse than handing back the point that was reached.
  //
  // On, and this one is a closer call than the periodic polish, so here is the
  // whole measurement. Across the eighteen instances it is adopted never at a
  // single tolerance and once - pilot87 - at a relaxed gap, and it moves the
  // totals the wrong way: 0.79x becomes 0.82x, 1.00x becomes 1.03x.
  //
  // It stays on because of where that cost lands. This only runs when the loop
  // failed to converge, so all sixteen instances that solve pay exactly
  // nothing; the entire 3% is greenbea and pilot87 spending a tenth of a budget
  // they had already exhausted. What it bought on pilot87 was a feasibility
  // violation of 1.8e-04 in place of 1.6e-02. Half the runs it can affect get
  // an answer they can use, and the other half were out of budget regardless.
  bool polish_on_exit = true;

  // The four things cuPDLPx adds on top of restarted Halpern PDHG, which is
  // what `halpern` above already is. Lu and Applegate, "cuPDLPx: A Further
  // Enhanced GPU-Based First-Order Solver for Linear Programming"
  // (arXiv:2507.14051). They report 2.5x-3.6x over cuPDLP on MIPLIB relaxations
  // and up to 6.8x on Mittelmann's set, and none of it needs a new matrix
  // product.
  //
  // All four are on, and here is what that rests on. Eighteen Netlib instances
  // with published optima, presolved, at 1e-8, against the Halpern base:
  //
  //   all four                      0.79x iterations, 0.90x time, 16/18 better
  //
  // Leave one out and the rest stay on:
  //
  //   without reflection            1.06x   stocfor1 stops solving
  //   without the constant step     3.21x   eight regress, maros-r7 is lost
  //   without the fixed-point rule  0.85x   stocfor1 stops solving
  //   without the PID weight        0.83x
  //
  // So none of the four is carrying the others, and the two large ones are
  // coupled: reflection overshoots the operator, and the adaptive rule's safety
  // condition was derived for a step that does not overshoot. Turn reflection
  // on while the step size still adapts and the iterates walk off - that is the
  // 3.21x, and maros-r7 running out of iterations inside it.
  //
  // Measured one at a time on 25fv47, each alone is neutral or worse than the
  // base: reflection and the PID weight both run to the iteration limit. Four
  // changes that only pay together are exactly the case for landing them
  // together and ablating afterwards.
  //
  // On the fifteen instances that actually finish - greenbea and pilot87 hit
  // the iteration limit either way and own most of the clock - the wall time
  // ratio is 0.57x.

  // Reflection: use R(z) = (1 + gamma) T(z) - gamma z in place of T(z), so the
  // step overshoots the operator's output rather than stopping at it. gamma = 0
  // is plain Halpern, gamma = 1 is Peaceman-Rachford.
  //
  // Free, as it happens. R(z) = T(z) + gamma (T(z) - z), and T(z) - z is the
  // dx and dy the step already produced; K R(z) = K T(z) + gamma K dx, and the
  // dual step already computes K dx to feed the adaptive step size rule. Three
  // vector updates, no product.
  double reflection = 1.0;

  // Fix the step size at `constant_step_scale / ||K||` instead of adapting it.
  // cuPDLPx does this and reports it as an improvement, which is worth
  // stating plainly: the adaptive rule is not just unnecessary there, it is
  // costing something. Every rejected trial is a matrix product spent on a step
  // that gets thrown away.
  bool constant_step_size = true;

  // The paper uses 0.998. This is 0.90, and that is measured: over all
  // eighty-eight Netlib instances at --tol=1e-8, against the iteration count on
  // the eighteen with published optima,
  //
  //   scale     reached the optimum     iterations
  //   off              76/88              1.27x
  //   0.90             75/88              1.06x
  //   0.95             74/88              1.02x
  //   0.998            74/88              1.00x
  //
  // A gentler step is more robust and slower, smoothly, and 0.90 buys back one
  // of the two instances the constant step costs for six per cent. 0.95 buys
  // nothing over 0.998 and is not worth its two per cent.
  //
  // The whole family still sits one instance below turning the cuPDLPx
  // additions off, which is the honest cost of a 1.27x speed gain and is
  // recorded here rather than buried.
  double constant_step_scale = 0.90;

  // Restart on the fixed-point residual ||z - T(z)|| rather than on the KKT
  // error, keeping the same three conditions and constants. The quantity is
  // already computed every iteration for the step size rule, so this restart
  // test is free where the KKT one costs two matrix products.
  bool restart_on_fixed_point = true;

  // PID control on the primal weight, replacing the exponential smoothing in
  // log space. The error signal is the primal-dual imbalance over the epoch,
  //
  //     e = log( sqrt(w) ||x - x_epoch|| / ( ||y - y_epoch|| / sqrt(w) ) )
  //     log w <- log w - [ Kp e + Ki sum(e) + Kd (e - e_previous) ]
  //
  // The smoothing this replaces is the same law with Kp = 0.5 and Ki = Kd = 0,
  // so the defaults below reproduce the old behaviour exactly and the two extra
  // terms can be measured one at a time. cuPDLPx does not publish its
  // coefficients, so these are ours to find.
  // cuPDLPx does not publish its coefficients, so these were swept here. Over
  // sixteen Netlib instances with published optima, presolved, at 1e-8,
  // measured in total iterations against Kp = 0.5 with the other two at zero -
  // which is the exponential smoothing this replaces:
  //
  //   Ki  0.02  1.039     Kd  0.1  0.968      Kp  0.3  1.031
  //       0.05  1.138         0.2  0.943          0.4  0.945 (with Kd 0.2)
  //       0.1   1.077         0.3  0.893          0.5  1.000
  //       0.2   1.321         0.4  0.910          0.6  0.945 (with Kd 0.2)
  //                           0.5  1.043          0.7  1.010
  //                           0.7  1.616
  //
  // The integral term hurts monotonically and is left at zero. The derivative
  // term has a clean minimum at 0.3 and turns hard on both sides of it. That
  // shape is what an integral term would be expected to cause trouble with
  // here: the primal weight is not tracking a fixed setpoint, it is chasing a
  // ratio that legitimately moves as the solve progresses, so accumulated error
  // is mostly stale information.
  bool pid_primal_weight = true;
  double primal_weight_kp = 0.5;
  double primal_weight_ki = 0.0;
  double primal_weight_kd = 0.3;

  ScalingOptions scaling;

  // Null means whatever this build and machine offer - CUDA when both are
  // present, plain C++ otherwise. Set it explicitly to force one, which is what
  // the CPU-versus-GPU comparison does.
  const class LinAlgBackend* backend = nullptr;

  // Look for a Farkas certificate of primal infeasibility. A first-order method
  // cannot otherwise tell an infeasible problem from a slowly converging one, so
  // it burns its whole budget and the caller has to guess. That is tolerable for
  // a standalone solve and ruinous inside branch and bound, where most nodes are
  // infeasible and a guess makes the dual bound worthless as a proof.
  bool detect_infeasibility = true;
  double infeasibility_tolerance = 1e-8;

  // Which of the three convergence measures have to hold to stop. The full test
  // is what a caller wants; the one-sided ones are how the polishing
  // sub-problems above are asked for exactly the half they are being solved
  // for, and nothing more.
  enum class Termination { kFull, kPrimalFeasibility, kDualFeasibility };
  Termination termination = Termination::kFull;

  // Start from this point rather than from zero. In the *scaled* space when
  // `scaling.enabled` is false, which is how the polishing sub-solves pass the
  // iterate that triggered them without a scaling round trip. Null means the
  // usual cold start.
  const std::vector<double>* warm_x = nullptr;
  const std::vector<double>* warm_y = nullptr;

  // Skip the power iteration and use this for ||K||. Only safe when the matrix
  // really is the one the number was measured on - the sub-solves reuse the
  // main solve's estimate, since they share its matrix exactly.
  double known_matrix_norm = 0.0;

  // Start from this step size and primal weight instead of deriving them from
  // the data. Algorithm 4 hands the sub-solves "the initial primal weight and
  // step size equal to the current values from Step 1", and it has to: the
  // sub-problems have had their c or their q zeroed, so the usual
  // ||c|| / ||q|| rule would hand back a weight derived from a vector that is
  // no longer there. Zero means derive it as usual.
  double initial_step_size = 0.0;
  double initial_primal_weight = 0.0;

  bool verbose = false;
  Int log_frequency = 5000;
};

enum class PdhgStatus {
  kOptimal,
  kPrimalInfeasible,
  kIterationLimit,
  kTimeLimit,
  kNumericalError,
};

std::string to_string(PdhgStatus status);

// Convergence measures, always reported for the unscaled problem, because that
// is the problem the user asked about.
struct PdhgResidual {
  double primal_residual = 0.0;  // 2-norm
  double dual_residual = 0.0;    // 2-norm

  // Largest single violation, unnormalised. Worth reporting separately because
  // the relative 2-norm measure can hide it: on a model with a hundred thousand
  // rows and a right-hand side in the thousands, dividing by ||q||_2 turns an
  // absolute violation of 0.07 into a "relative residual" of 5e-06. That is
  // exactly how a 3.6% objective error passed a 1e-4 tolerance on
  // irish-electricity.
  double primal_residual_inf = 0.0;
  double dual_residual_inf = 0.0;
  double primal_objective = 0.0;
  double dual_objective = 0.0;
  double absolute_gap = 0.0;
  double relative_primal = 0.0;
  double relative_dual = 0.0;
  double relative_gap = 0.0;

  // The same two residuals measured in the infinity norm and normalised by the
  // infinity norm of the data. PDLP uses the 2-norm form above; HiGHS and OSQP
  // use this one. The difference matters on tall problems: a 2-norm residual
  // divided by ||q||_2 accumulates the right-hand side over a hundred thousand
  // rows, so a single genuinely violated constraint disappears into the
  // denominator. The infinity-norm form cannot hide one bad row.
  double relative_primal_inf = 0.0;
  double relative_dual_inf = 0.0;

  // `absolute_tolerance` of 0 keeps the pure PDLP behaviour; anything positive
  // additionally demands that no single row or reduced cost is out by more than
  // that much, which is what HiGHS and OSQP do.
  bool primal_ok(double tolerance, double absolute_tolerance = 0.0,
                 bool require_inf_norm = true) const {
    bool ok = relative_primal <= tolerance;
    if (require_inf_norm) ok = ok && relative_primal_inf <= tolerance;
    if (absolute_tolerance > 0.0) ok = ok && primal_residual_inf <= absolute_tolerance;
    return ok;
  }

  bool dual_ok(double tolerance, double absolute_tolerance = 0.0,
               bool require_inf_norm = true) const {
    bool ok = relative_dual <= tolerance;
    if (require_inf_norm) ok = ok && relative_dual_inf <= tolerance;
    if (absolute_tolerance > 0.0) ok = ok && dual_residual_inf <= absolute_tolerance;
    return ok;
  }

  bool gap_ok(double tolerance) const { return relative_gap <= tolerance; }

  // `gap_tolerance` of zero means the gap is held to `tolerance` like
  // everything else, which is the ordinary case.
  bool converged(double tolerance, double absolute_tolerance = 0.0,
                 bool require_inf_norm = true, double gap_tolerance = 0.0) const {
    return primal_ok(tolerance, absolute_tolerance, require_inf_norm) &&
           dual_ok(tolerance, absolute_tolerance, require_inf_norm) &&
           gap_ok(gap_tolerance > 0.0 ? gap_tolerance : tolerance);
  }

  // How far this point is from passing, as one number: the largest of the
  // relative measures. The polish trigger compares it against the target.
  double worst_relative(bool require_inf_norm = true) const {
    double worst = std::fmax(relative_primal, std::fmax(relative_dual, relative_gap));
    if (require_inf_norm) {
      worst = std::fmax(worst, std::fmax(relative_primal_inf, relative_dual_inf));
    }
    return worst;
  }
};

struct PdhgResult {
  PdhgStatus status = PdhgStatus::kNumericalError;
  std::vector<double> x;  // primal solution, unscaled
  std::vector<double> y;  // dual solution, unscaled
  double objective = 0.0;  // in the original model's sense, including its offset

  Int iterations = 0;
  Int restarts = 0;

  // Polishing, reported so a run can be read for whether it earned its keep.
  Int polish_attempts = 0;
  Int polish_iterations = 0;  // included in `iterations`
  bool polished = false;      // the returned point came out of a polish
  double solve_seconds = 0.0;
  double matrix_norm_estimate = 0.0;
  double final_step_size = 0.0;
  double final_primal_weight = 0.0;

  PdhgResidual residual;
  ScalingReport scaling;
  std::string message;
};

PdhgResult solve_pdhg(const StandardLp& lp, const PdhgOptions& options = {});

// Evaluates the convergence measures of a primal-dual pair against an LP. Public
// because the tests and the benchmark harness both check solutions that this
// solver did not produce.
PdhgResidual evaluate_residual(const StandardLp& lp, const std::vector<double>& x,
                               const std::vector<double>& y);

// Largest singular value of K, by power iteration on K'K. Returns 0 for an empty
// matrix.
double estimate_matrix_norm(const StandardLp& lp, int iterations, double tolerance);

}  // namespace sankhya
