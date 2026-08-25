#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

// Convex quadratic programming by ADMM, in the operator-splitting form of OSQP.
//
//   minimise  (1/2) x' P x + q' x
//   subject to      l <= A x <= u
//
// Variable bounds are folded in as rows of A, which is what lets the whole
// feasible set be one box in the z variable and keeps the projection a clamp.
//
// The linear system at the heart of each iteration is solved indirectly, by
// conjugate gradient, rather than by factorising the quasi-definite KKT matrix.
// That is a deliberate trade. A direct solve is faster per iteration on a CPU,
// but it needs a sparse LDL' with a fill-reducing ordering - a week of work that
// does not exist yet - and it does not move to a GPU. The indirect form needs
// only sparse matrix-vector products, which means it runs on exactly the kernels
// the linear programming path already has, on either backend.
struct QpOptions {
  double absolute_tolerance = 1e-6;
  double relative_tolerance = 1e-6;

  // A hard ceiling on the residuals, on top of the relative test above.
  //
  // The relative dual tolerance is eps_abs + eps_rel * max(||Px||, ||A'y||,
  // ||q||), which inflates as the dual iterate grows. On the dualc family from
  // Maros-Meszaros that let a dual residual of 2.6e-02 pass a request for 1e-8
  // and report optimal on an objective that was 0.9% wrong. Same failure the
  // linear programming path had, in a different normalisation.
  //
  // Zero restores plain OSQP behaviour. A positive value refuses to call a point
  // optimal when a residual is genuinely large, whatever the normalisation says.
  double max_absolute_residual = 1e-4;

  Int max_iterations = 20000;
  double time_limit_seconds = 300.0;
  Int termination_check_frequency = 25;

  // OSQP's defaults, from its paper.
  double sigma = 1e-6;   // primal regularisation, keeps the system solvable
  double alpha = 1.6;    // over-relaxation
  double rho = 0.1;      // initial step size
  double equality_rho_multiplier = 1e3;  // equalities are active at the optimum

  bool adaptive_rho = true;
  double rho_update_threshold = 5.0;  // only update on a change this large

  // Conjugate gradient budget per ADMM iteration. Loose early, tightened as the
  // outer iteration converges.
  Int cg_max_iterations = 200;
  double cg_tolerance = 1e-7;

  bool scaling = true;
  Int ruiz_iterations = 10;

  bool verbose = false;
  Int log_frequency = 500;
};

enum class QpStatus {
  kOptimal,
  kIterationLimit,
  kTimeLimit,
  kNumericalError,
};

std::string to_string(QpStatus status);

struct QpResidual {
  double primal = 0.0;  // ||Ax - z||_inf
  double dual = 0.0;    // ||Px + q + A'y||_inf
  double primal_tolerance = 0.0;
  double dual_tolerance = 0.0;
  double absolute_cap = 0.0;
  bool converged() const {
    const bool relative_ok = primal <= primal_tolerance && dual <= dual_tolerance;
    if (absolute_cap <= 0.0) return relative_ok;
    return relative_ok && primal <= absolute_cap && dual <= absolute_cap;
  }
};

struct QpResult {
  QpStatus status = QpStatus::kNumericalError;
  std::vector<double> x;
  std::vector<double> y;
  double objective = 0.0;

  Int iterations = 0;
  Int cg_iterations = 0;
  Int rho_updates = 0;
  double solve_seconds = 0.0;
  double final_rho = 0.0;
  QpResidual residual;
  std::string message;
};

QpResult solve_qp(const Model& model, const QpOptions& options = {});

}  // namespace sankhya
