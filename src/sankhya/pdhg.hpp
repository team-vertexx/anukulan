#pragma once

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
  bool converged(double tolerance, double absolute_tolerance = 0.0,
                 bool require_inf_norm = true) const {
    bool ok = relative_primal <= tolerance && relative_dual <= tolerance &&
              relative_gap <= tolerance;
    if (require_inf_norm) {
      ok = ok && relative_primal_inf <= tolerance && relative_dual_inf <= tolerance;
    }
    if (absolute_tolerance > 0.0) {
      ok = ok && primal_residual_inf <= absolute_tolerance &&
           dual_residual_inf <= absolute_tolerance;
    }
    return ok;
  }
};

struct PdhgResult {
  PdhgStatus status = PdhgStatus::kNumericalError;
  std::vector<double> x;  // primal solution, unscaled
  std::vector<double> y;  // dual solution, unscaled
  double objective = 0.0;  // in the original model's sense, including its offset

  Int iterations = 0;
  Int restarts = 0;
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
