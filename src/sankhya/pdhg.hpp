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

  // Each of these is a separate piece of machinery, kept switchable so their
  // effect can be measured one at a time rather than asserted.
  bool adaptive_step_size = true;
  bool primal_weight_updates = true;
  bool restarts = true;

  ScalingOptions scaling;

  bool verbose = false;
  Int log_frequency = 5000;
};

enum class PdhgStatus {
  kOptimal,
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

  // `absolute_tolerance` of 0 keeps the pure PDLP behaviour; anything positive
  // additionally demands that no single row or reduced cost is out by more than
  // that much, which is what HiGHS and OSQP do.
  bool converged(double tolerance, double absolute_tolerance = 0.0) const {
    const bool relative_ok = relative_primal <= tolerance &&
                             relative_dual <= tolerance && relative_gap <= tolerance;
    if (absolute_tolerance <= 0.0) return relative_ok;
    return relative_ok && primal_residual_inf <= absolute_tolerance &&
           dual_residual_inf <= absolute_tolerance;
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
