#pragma once

#include <string>
#include <vector>

#include "sankhya/standard_form.hpp"

namespace sankhya {

// Diagonal preconditioning for the first-order method. The problem is rewritten
// in terms of a scaled variable x_tilde with x = D2 * x_tilde:
//
//   K_tilde = D1 K D2      q_tilde = D1 q       c_tilde = D2 c
//   lower_tilde = lower / d2                    upper_tilde = upper / d2
//
// and the dual maps back as y = D1 * y_tilde.
//
// Two schemes, applied in this order, following PDLP:
//   1. Ruiz equilibration in the infinity norm, a handful of sweeps
//   2. Pock-Chambolle with alpha = 1, which is the same idea in the 1-norm
//
// This is not a tuning knob. Netlib has instances whose coefficients span ten
// orders of magnitude, and PDHG on those without preconditioning does not
// converge in any useful number of iterations.
struct ScalingOptions {
  int ruiz_iterations = 10;
  bool pock_chambolle = true;
};

struct Scaling {
  std::vector<double> row_scale;  // D1 diagonal
  std::vector<double> col_scale;  // D2 diagonal

  // x = D2 * x_tilde
  void unscale_primal(const std::vector<double>& scaled, std::vector<double>* out) const;
  // y = D1 * y_tilde
  void unscale_dual(const std::vector<double>& scaled, std::vector<double>* out) const;
};

struct ScalingReport {
  Scaling scaling;
  // Ratio of largest to smallest nonzero row (and column) infinity norm, before
  // and after. This is the number the robustness slide is built on.
  double row_spread_before = 1.0;
  double row_spread_after = 1.0;
  double col_spread_before = 1.0;
  double col_spread_after = 1.0;
};

ScalingReport scale_lp(StandardLp* lp, const ScalingOptions& options = {});

}  // namespace sankhya
