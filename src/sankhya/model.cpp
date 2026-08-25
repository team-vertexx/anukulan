#include "sankhya/model.hpp"

#include <cmath>
#include <cstdio>

namespace sankhya {

bool Model::has_integers() const {
  for (const VarType t : col_type) {
    if (t != VarType::kContinuous) return true;
  }
  return false;
}

bool Model::validate(std::string* error) const {
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };

  if (!constraints.validate(error)) return false;

  const std::size_t n = sz(num_cols());
  const std::size_t m = sz(num_rows());
  if (objective.size() != n) return fail("objective length does not match columns");
  if (col_lower.size() != n || col_upper.size() != n)
    return fail("column bound length does not match columns");
  if (col_type.size() != n) return fail("column type length does not match columns");
  if (row_lower.size() != m || row_upper.size() != m)
    return fail("row bound length does not match rows");
  if (!row_names.empty() && row_names.size() != m)
    return fail("row name length does not match rows");
  if (!col_names.empty() && col_names.size() != n)
    return fail("column name length does not match columns");

  for (std::size_t j = 0; j < n; ++j) {
    if (std::isnan(col_lower[j]) || std::isnan(col_upper[j]))
      return fail("NaN column bound at column " + std::to_string(j));
    if (col_lower[j] > col_upper[j])
      return fail("empty column bound interval at column " + std::to_string(j));
    if (std::isnan(objective[j]))
      return fail("NaN objective coefficient at column " + std::to_string(j));
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (std::isnan(row_lower[i]) || std::isnan(row_upper[i]))
      return fail("NaN row bound at row " + std::to_string(i));
    if (row_lower[i] > row_upper[i])
      return fail("empty row bound interval at row " + std::to_string(i));
  }

  if (hessian.nnz() > 0) {
    if (hessian.rows() != num_cols() || hessian.cols() != num_cols())
      return fail("hessian is not square in the number of columns");
    // Q must be symmetric: comparing against its own transpose catches both a
    // missing mirrored entry and a mismatched value.
    const SparseMatrix qt = hessian.transpose();
    if (qt.start() != hessian.start() || qt.index() != hessian.index())
      return fail("hessian sparsity pattern is not symmetric");
    for (std::size_t k = 0; k < hessian.value().size(); ++k) {
      if (hessian.value()[k] != qt.value()[k]) return fail("hessian is not symmetric");
    }
  }
  return true;
}

ModelViolation measure_violation(const Model& model, const std::vector<double>& x) {
  ModelViolation out;
  if (static_cast<Int>(x.size()) != model.num_cols()) return out;

  for (Int j = 0; j < model.num_cols(); ++j) {
    const double v = x[sz(j)];
    const double below = model.col_lower[sz(j)] - v;
    const double above = v - model.col_upper[sz(j)];
    const double miss = std::fmax(std::isfinite(below) ? below : 0.0,
                                  std::isfinite(above) ? above : 0.0);
    if (miss > out.bound_violation) {
      out.bound_violation = miss;
      out.worst_col = j;
    }
  }

  std::vector<double> activity(sz(model.num_rows()), 0.0);
  if (model.num_rows() > 0) model.constraints.multiply(x.data(), activity.data());
  for (Int i = 0; i < model.num_rows(); ++i) {
    const double a = activity[sz(i)];
    const double below = model.row_lower[sz(i)] - a;
    const double above = a - model.row_upper[sz(i)];
    const double miss = std::fmax(std::isfinite(below) ? below : 0.0,
                                  std::isfinite(above) ? above : 0.0);
    if (miss > out.row_violation) {
      out.row_violation = miss;
      out.worst_row = i;
    }
  }
  return out;
}

ModelStats compute_stats(const Model& model) {
  ModelStats s;
  s.rows = model.num_rows();
  s.cols = model.num_cols();
  s.nnz = model.constraints.nnz();
  s.hessian_nnz = model.hessian.nnz();

  for (std::size_t i = 0; i < model.row_lower.size(); ++i) {
    const bool lo = model.row_lower[i] > -kInf;
    const bool hi = model.row_upper[i] < kInf;
    if (lo && hi) {
      if (model.row_lower[i] == model.row_upper[i]) {
        s.equality_rows++;
      } else {
        s.range_rows++;
      }
    } else if (hi) {
      s.less_rows++;
    } else if (lo) {
      s.greater_rows++;
    } else {
      s.free_rows++;
    }
  }

  for (std::size_t j = 0; j < model.col_lower.size(); ++j) {
    const bool lo = model.col_lower[j] > -kInf;
    const bool hi = model.col_upper[j] < kInf;
    const bool integral = model.col_type[j] != VarType::kContinuous;
    if (integral) {
      s.integer_cols++;
      if (model.col_lower[j] == 0.0 && model.col_upper[j] == 1.0) s.binary_cols++;
    }
    if (!lo && !hi) s.free_cols++;
    if (lo && hi) {
      s.boxed_cols++;
      if (model.col_lower[j] == model.col_upper[j]) s.fixed_cols++;
    }
  }

  double lo_mag = kInf;
  double hi_mag = 0.0;
  for (const double v : model.constraints.value()) {
    const double a = std::fabs(v);
    if (a < lo_mag) lo_mag = a;
    if (a > hi_mag) hi_mag = a;
  }
  if (s.nnz > 0) {
    s.min_abs_coeff = lo_mag;
    s.max_abs_coeff = hi_mag;
    s.coeff_dynamic_range = (lo_mag > 0.0) ? hi_mag / lo_mag : kInf;
  }
  return s;
}

std::string format_stats(const Model& model, const ModelStats& s) {
  char buf[2048];
  const double density =
      (s.rows > 0 && s.cols > 0)
          ? 100.0 * static_cast<double>(s.nnz) /
                (static_cast<double>(s.rows) * static_cast<double>(s.cols))
          : 0.0;
  std::snprintf(
      buf, sizeof(buf),
      "name          %s\n"
      "sense         %s\n"
      "rows          %d  (eq %d, <= %d, >= %d, range %d, free %d)\n"
      "cols          %d  (integer %d, binary %d, free %d, boxed %d, fixed %d)\n"
      "nonzeros      %d  (%.4f%% dense)\n"
      "hessian nnz   %d\n"
      "|a| range     %.3e .. %.3e  (ratio %.3e)\n"
      "obj offset    %.10e\n",
      model.name.c_str(),
      model.sense == ObjSense::kMinimize ? "minimize" : "maximize", s.rows,
      s.equality_rows, s.less_rows, s.greater_rows, s.range_rows, s.free_rows, s.cols,
      s.integer_cols, s.binary_cols, s.free_cols, s.boxed_cols, s.fixed_cols, s.nnz,
      density, s.hessian_nnz, s.min_abs_coeff, s.max_abs_coeff, s.coeff_dynamic_range,
      model.objective_offset);
  return std::string(buf);
}

}  // namespace sankhya
