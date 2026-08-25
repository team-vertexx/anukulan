#pragma once

#include <string>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

enum class ObjSense { kMinimize, kMaximize };

enum class VarType { kContinuous, kInteger, kSemiContinuous, kSemiInteger };

// A model in its natural file form:
//
//   optimize   0.5 x' Q x + c' x + offset
//   subject to row_lower <= A x <= row_upper
//              col_lower <= x   <= col_upper
//
// Q is empty for an LP and is stored in full symmetric form. This is deliberately
// the same shape HiGHS uses, because HiGHS is what the results get checked against;
// the solver-specific standard forms are derived from this, not stored instead of it.
struct Model {
  std::string name;
  ObjSense sense = ObjSense::kMinimize;
  double objective_offset = 0.0;

  std::vector<double> objective;  // c, length num_cols()
  SparseMatrix constraints;       // A, num_rows() x num_cols()
  SparseMatrix hessian;           // Q, num_cols() x num_cols(), full symmetric

  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<VarType> col_type;

  std::vector<std::string> row_names;
  std::vector<std::string> col_names;

  Int num_rows() const { return constraints.rows(); }
  Int num_cols() const { return constraints.cols(); }
  bool has_hessian() const { return hessian.nnz() > 0; }
  bool has_integers() const;

  // Checks internal consistency: matching lengths, lower <= upper, finite where it
  // must be, symmetric Q. Returns false and fills `error` on the first problem.
  bool validate(std::string* error) const;
};

struct ModelStats {
  Int rows = 0;
  Int cols = 0;
  Int nnz = 0;
  Int hessian_nnz = 0;

  Int equality_rows = 0;
  Int less_rows = 0;     // finite upper only
  Int greater_rows = 0;  // finite lower only
  Int range_rows = 0;    // both finite and different
  Int free_rows = 0;     // neither finite

  Int integer_cols = 0;
  Int binary_cols = 0;
  Int free_cols = 0;
  Int boxed_cols = 0;  // both bounds finite
  Int fixed_cols = 0;

  // Magnitudes of the nonzeros in A. Useful as a first look at conditioning.
  double min_abs_coeff = 0.0;
  double max_abs_coeff = 0.0;
  double coeff_dynamic_range = 0.0;  // max / min, 0 when the matrix is empty
};

ModelStats compute_stats(const Model& model);

// How badly a point misses the model it is supposed to solve.
//
// This exists because of what presolve did to the safety net. The solver's
// absolute residual cap is measured on whatever it was handed, and once
// presolve is in front of it that is the reduced model - whose rows have been
// removed, merged and rescaled. On Netlib's share1b the cap brought the worst
// row violation of an unpresolved solve from 2.9e-03 down to 8.7e-09, and made
// no difference at all to a presolved one, which stayed at 4.2e-02. The check
// was not failing; it was passing, on the wrong object.
//
// So the claim gets verified against the model the caller actually handed in,
// after postsolve, by walking its own rows and its own bounds.
struct ModelViolation {
  double row_violation = 0.0;    // largest absolute amount a row bound is missed by
  double bound_violation = 0.0;  // largest absolute amount a column bound is missed by
  Int worst_row = -1;
  Int worst_col = -1;

  double worst() const {
    return row_violation > bound_violation ? row_violation : bound_violation;
  }
};

ModelViolation measure_violation(const Model& model, const std::vector<double>& x);

std::string format_stats(const Model& model, const ModelStats& stats);

}  // namespace sankhya
