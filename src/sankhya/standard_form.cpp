#include "sankhya/standard_form.hpp"

#include <cmath>

namespace sankhya {

double StandardLp::standard_objective(const std::vector<double>& x) const {
  double sum = 0.0;
  for (std::size_t j = 0; j < c.size(); ++j) sum += c[j] * x[j];
  return sum;
}

double StandardLp::model_objective(const std::vector<double>& x) const {
  return objective_scale * standard_objective(x) + objective_offset;
}

void StandardLp::primal_residual(const std::vector<double>& x,
                                 std::vector<double>* scratch, double* two_norm,
                                 double* inf_norm) const {
  scratch->resize(sz(k.rows()));
  k.multiply(x.data(), scratch->data());

  double sum_sq = 0.0;
  double worst = 0.0;
  for (Int i = 0; i < k.rows(); ++i) {
    const double row = (*scratch)[sz(i)];
    // Equality rows are violated in either direction; inequality rows only when
    // the activity falls below the right-hand side.
    const double violation =
        (i < num_equalities) ? (row - q[sz(i)]) : std::fmin(row - q[sz(i)], 0.0);
    sum_sq += violation * violation;
    worst = std::fmax(worst, std::fabs(violation));
  }
  if (two_norm) *two_norm = std::sqrt(sum_sq);
  if (inf_norm) *inf_norm = worst;
}

bool StandardLp::validate(std::string* error) const {
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };
  if (!k.validate(error)) return false;
  if (c.size() != sz(k.cols())) return fail("objective length does not match columns");
  if (q.size() != sz(k.rows())) return fail("rhs length does not match rows");
  if (lower.size() != sz(k.cols()) || upper.size() != sz(k.cols()))
    return fail("bound length does not match columns");
  if (row_origin.size() != sz(k.rows()))
    return fail("row origin length does not match rows");
  if (num_equalities < 0 || num_equalities > k.rows())
    return fail("num_equalities is out of range");
  for (std::size_t j = 0; j < lower.size(); ++j) {
    if (lower[j] > upper[j])
      return fail("empty bound interval at column " + std::to_string(j));
  }
  for (const double v : q) {
    if (std::isnan(v) || std::isinf(v)) return fail("rhs is not finite");
  }

  if (kt.rows() != k.cols() || kt.cols() != k.rows())
    return fail("cached transpose has the wrong shape");
  if (kt.nnz() != k.nnz()) return fail("cached transpose has a different nnz");
  return true;
}

StandardFormResult to_standard_form(const Model& model) {
  StandardFormResult result;
  StandardLp& lp = result.lp;

  const Int n = model.num_cols();
  const Int m = model.num_rows();

  lp.objective_scale = (model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
  lp.objective_offset = model.objective_offset;
  lp.c.resize(sz(n));
  for (Int j = 0; j < n; ++j) lp.c[sz(j)] = lp.objective_scale * model.objective[sz(j)];
  lp.lower = model.col_lower;
  lp.upper = model.col_upper;

  // Two passes over the rows: equalities are collected first so they occupy the
  // leading block of K, then everything else.
  struct Emit {
    Int model_row;
    double sign;
    double rhs;
  };
  std::vector<Emit> equalities;
  std::vector<Emit> inequalities;

  for (Int i = 0; i < m; ++i) {
    const double lo = model.row_lower[sz(i)];
    const double hi = model.row_upper[sz(i)];
    const bool has_lo = lo > -kInf;
    const bool has_hi = hi < kInf;

    if (has_lo && has_hi && lo == hi) {
      equalities.push_back(Emit{i, 1.0, lo});
      result.rows_from_equalities++;
    } else if (has_lo && has_hi) {
      // Two-sided row: a'x >= lo and -a'x >= -hi.
      inequalities.push_back(Emit{i, 1.0, lo});
      inequalities.push_back(Emit{i, -1.0, -hi});
      result.rows_from_ranges++;
    } else if (has_lo) {
      inequalities.push_back(Emit{i, 1.0, lo});
      result.rows_from_inequalities++;
    } else if (has_hi) {
      inequalities.push_back(Emit{i, -1.0, -hi});
      result.rows_from_inequalities++;
    } else {
      // A row with no finite bound constrains nothing.
      result.free_rows_dropped++;
    }
  }

  const Int num_rows =
      static_cast<Int>(equalities.size()) + static_cast<Int>(inequalities.size());
  lp.num_equalities = static_cast<Int>(equalities.size());
  lp.q.resize(sz(num_rows));
  lp.row_origin.resize(sz(num_rows));

  std::vector<Triplet> entries;
  entries.reserve(sz(model.constraints.nnz()) + sz(result.rows_from_ranges) * 4);

  auto emit_rows = [&](const std::vector<Emit>& list, Int offset) {
    for (std::size_t e = 0; e < list.size(); ++e) {
      const Emit& item = list[e];
      const Int dest = offset + static_cast<Int>(e);
      lp.q[sz(dest)] = item.rhs;
      lp.row_origin[sz(dest)] = StandardLp::RowOrigin{item.model_row, item.sign};
      const Int begin = model.constraints.row_begin(item.model_row);
      const Int end = model.constraints.row_end(item.model_row);
      for (Int k = begin; k < end; ++k) {
        entries.push_back(Triplet{dest, model.constraints.index()[sz(k)],
                                  item.sign * model.constraints.value()[sz(k)]});
      }
    }
  };
  emit_rows(equalities, 0);
  emit_rows(inequalities, lp.num_equalities);

  lp.k = SparseMatrix::from_triplets(num_rows, n, std::move(entries));
  lp.kt = lp.k.transpose();

  if (result.free_rows_dropped > 0) {
    result.warnings.push_back(
        std::to_string(result.free_rows_dropped) +
        " row(s) with no finite bound dropped: they constrain nothing");
  }

  std::string error;
  if (!lp.validate(&error)) {
    result.ok = false;
    result.error = "standard form failed validation: " + error;
    return result;
  }
  result.ok = true;
  return result;
}

}  // namespace sankhya
