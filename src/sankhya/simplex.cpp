#include "sankhya/simplex.hpp"

#include <algorithm>
#include <cmath>

namespace sankhya {

LogicalForm to_logical_form(const StandardLp& lp) {
  LogicalForm form;
  form.num_structural = lp.num_cols();
  form.num_rows = lp.num_rows();
  form.num_equalities = lp.num_equalities;
  form.objective_scale = lp.objective_scale;
  form.objective_offset = lp.objective_offset;

  const Int slacks = lp.num_rows() - lp.num_equalities;
  const Int columns = form.num_structural + slacks;

  form.cost.assign(sz(columns), 0.0);
  for (Int j = 0; j < form.num_structural; ++j) form.cost[sz(j)] = lp.c[sz(j)];
  form.rhs = lp.q;
  form.lower.assign(sz(columns), 0.0);
  form.upper.assign(sz(columns), kInf);
  for (Int j = 0; j < form.num_structural; ++j) {
    form.lower[sz(j)] = lp.lower[sz(j)];
    form.upper[sz(j)] = lp.upper[sz(j)];
  }

  // Inequality row i reads a'x >= q, which becomes a'x - s = q with s >= 0.
  std::vector<Triplet> entries;
  entries.reserve(sz(lp.k.nnz()) + sz(slacks));
  for (Int j = 0; j < form.num_structural; ++j) {
    for (Int e = lp.kt.row_begin(j); e < lp.kt.row_end(j); ++e) {
      entries.push_back(Triplet{j, lp.kt.index()[sz(e)], lp.kt.value()[sz(e)]});
    }
  }
  for (Int i = lp.num_equalities; i < lp.num_rows(); ++i) {
    const Int column = form.num_structural + (i - lp.num_equalities);
    entries.push_back(Triplet{column, i, -1.0});
  }
  form.columns = SparseMatrix::from_triplets(columns, lp.num_rows(), std::move(entries));
  form.rows = form.columns.transpose();
  return form;
}

bool SimplexBasis::set_initial(const LogicalForm& form, std::string* error) {
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();
  basic_.assign(sz(m), -1);
  status_.assign(sz(columns), VarStatus::kAtLower);

  for (Int j = 0; j < columns; ++j) {
    if (form.lower[sz(j)] <= -kInf && form.upper[sz(j)] >= kInf) {
      status_[sz(j)] = VarStatus::kFree;
    } else if (form.lower[sz(j)] <= -kInf) {
      status_[sz(j)] = VarStatus::kAtUpper;
    }
  }

  // Slacks are already unit columns, so they take their own rows.
  for (Int i = form.num_equalities; i < m; ++i) {
    const Int slack = form.num_structural + (i - form.num_equalities);
    basic_[sz(i)] = slack;
    status_[sz(slack)] = VarStatus::kBasic;
  }

  // Equality rows need a structural column each. Prefer one that appears in few
  // rows, so the starting basis is as close to triangular as it can be.
  std::vector<bool> taken(sz(columns), false);
  for (Int i = 0; i < form.num_equalities; ++i) {
    Int best = -1;
    Int best_length = 0;
    double best_magnitude = 0.0;
    for (Int e = form.rows.row_begin(i); e < form.rows.row_end(i); ++e) {
      const Int j = form.rows.index()[sz(e)];
      if (taken[sz(j)] || status_[sz(j)] == VarStatus::kBasic) continue;
      const double magnitude = std::fabs(form.rows.value()[sz(e)]);
      if (magnitude < 1e-7) continue;
      const Int length = form.columns.row_end(j) - form.columns.row_begin(j);
      if (best < 0 || length < best_length ||
          (length == best_length && magnitude > best_magnitude)) {
        best = j;
        best_length = length;
        best_magnitude = magnitude;
      }
    }
    if (best < 0) {
      if (error) *error = "equality row " + std::to_string(i) + " has no usable column";
      return false;
    }
    taken[sz(best)] = true;
    basic_[sz(i)] = best;
    status_[sz(best)] = VarStatus::kBasic;
  }
  return refactorize(form, error);
}

bool SimplexBasis::refactorize(const LogicalForm& form, std::string* error) {
  updates_ = 0;
  return lu_.factorize(form.columns, basic_, LuOptions{}, error);
}

void SimplexBasis::compute_primal(const LogicalForm& form,
                                  std::vector<double>* values) const {
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();
  values->assign(sz(columns), 0.0);

  // Non-basic variables sit on a bound.
  for (Int j = 0; j < columns; ++j) {
    switch (status_[sz(j)]) {
      case VarStatus::kAtLower:
        (*values)[sz(j)] = std::isfinite(form.lower[sz(j)]) ? form.lower[sz(j)] : 0.0;
        break;
      case VarStatus::kAtUpper:
        (*values)[sz(j)] = std::isfinite(form.upper[sz(j)]) ? form.upper[sz(j)] : 0.0;
        break;
      case VarStatus::kFree:
        (*values)[sz(j)] = 0.0;
        break;
      case VarStatus::kBasic:
        break;
    }
  }

  // rhs minus the contribution of everything non-basic.
  std::vector<double> residual = form.rhs;
  for (Int j = 0; j < columns; ++j) {
    if (status_[sz(j)] == VarStatus::kBasic) continue;
    const double value = (*values)[sz(j)];
    if (value == 0.0) continue;
    for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
      residual[sz(form.columns.index()[sz(e)])] -= form.columns.value()[sz(e)] * value;
    }
  }

  lu_.ftran(&residual);
  for (Int i = 0; i < m; ++i) (*values)[sz(basic_[sz(i)])] = residual[sz(i)];
}

void SimplexBasis::compute_duals(const LogicalForm& form, std::vector<double>* duals,
                                 std::vector<double>* reduced_costs) const {
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();

  std::vector<double> cb(sz(m), 0.0);
  for (Int i = 0; i < m; ++i) cb[sz(i)] = form.cost[sz(basic_[sz(i)])];
  lu_.btran(&cb);
  *duals = cb;

  reduced_costs->assign(sz(columns), 0.0);
  for (Int j = 0; j < columns; ++j) {
    if (status_[sz(j)] == VarStatus::kBasic) continue;
    double dot = 0.0;
    for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
      dot += form.columns.value()[sz(e)] * cb[sz(form.columns.index()[sz(e)])];
    }
    (*reduced_costs)[sz(j)] = form.cost[sz(j)] - dot;
  }
}

void SimplexBasis::ftran_column(const LogicalForm& form, Int column,
                                std::vector<double>* out) const {
  out->assign(sz(form.num_rows), 0.0);
  for (Int e = form.columns.row_begin(column); e < form.columns.row_end(column); ++e) {
    (*out)[sz(form.columns.index()[sz(e)])] = form.columns.value()[sz(e)];
  }
  lu_.ftran(out);
}

bool SimplexBasis::pivot(const LogicalForm& form, Int leaving_row, Int entering,
                         VarStatus leaving_to, std::string* error) {
  const Int leaving = basic_[sz(leaving_row)];
  basic_[sz(leaving_row)] = entering;
  status_[sz(entering)] = VarStatus::kBasic;
  status_[sz(leaving)] = leaving_to;

  // No product-form update yet, so every pivot refactorises. That is correct and
  // slow; the update comes next, and this is the version whose answers the
  // update will be checked against.
  ++updates_;
  if (!lu_.factorize(form.columns, basic_, LuOptions{}, error)) {
    basic_[sz(leaving_row)] = leaving;
    status_[sz(leaving)] = VarStatus::kBasic;
    status_[sz(entering)] = leaving_to;
    return false;
  }
  return true;
}

}  // namespace sankhya
