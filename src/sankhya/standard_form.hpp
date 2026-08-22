#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

// The form the first-order solver works in:
//
//   min  c' x
//   s.t. K_eq   x  = q_eq      (the first `num_equalities` rows of K)
//        K_ineq x >= q_ineq    (the remaining rows)
//        lower <= x <= upper
//
// Equality rows come first so that the dual projection is a single split: leave
// the first block alone, clamp the rest at zero.
//
// This is the form used by PDLP and cuPDLP. A two-sided row is split into two
// inequality rows, which costs a duplicated row but keeps the dual projection
// trivial - worth it, since that projection runs every iteration on the GPU.
struct StandardLp {
  SparseMatrix k;   // stacked constraint matrix, (num_equalities + inequalities) x n
  SparseMatrix kt;  // K transpose, cached because PDHG needs both directions
  std::vector<double> c;
  std::vector<double> q;
  std::vector<double> lower;
  std::vector<double> upper;
  Int num_equalities = 0;

  // Recovering the model's own objective from a standard-form point:
  //   model objective = objective_scale * (c' x) + objective_offset
  // objective_scale is -1 when the model was a maximization, since c is negated
  // to turn it into a minimization.
  double objective_scale = 1.0;
  double objective_offset = 0.0;

  // Where each standard-form row came from, so row duals can be mapped back.
  struct RowOrigin {
    Int model_row = 0;
    double sign = 1.0;  // standard row = sign * model row
  };
  std::vector<RowOrigin> row_origin;

  Int num_rows() const { return k.rows(); }
  Int num_cols() const { return k.cols(); }
  Int num_inequalities() const { return k.rows() - num_equalities; }

  double standard_objective(const std::vector<double>& x) const;
  double model_objective(const std::vector<double>& x) const;

  // ||(K_eq x - q_eq, [q_ineq - K_ineq x]^+)||_2 and its infinity norm.
  // `scratch` must have num_rows() entries; it holds K x on return.
  void primal_residual(const std::vector<double>& x, std::vector<double>* scratch,
                       double* two_norm, double* inf_norm) const;

  bool validate(std::string* error) const;
};

struct StandardFormResult {
  bool ok = false;
  std::string error;
  std::vector<std::string> warnings;
  StandardLp lp;

  Int rows_from_equalities = 0;
  Int rows_from_inequalities = 0;
  Int rows_from_ranges = 0;  // two-sided rows, each contributing two rows
  Int free_rows_dropped = 0;
};

// Integrality is ignored here: this builds the continuous relaxation, which is
// exactly what branch and bound will want at every node later.
StandardFormResult to_standard_form(const Model& model);

}  // namespace sankhya
