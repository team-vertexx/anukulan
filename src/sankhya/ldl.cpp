#include "sankhya/ldl.hpp"

#include <cmath>
#include <string>

namespace sankhya {

bool LdlFactor::analyse(const SparseMatrix& lower, std::string* error) {
  if (lower.rows() != lower.cols()) {
    if (error) *error = "LDL' needs a square matrix";
    return false;
  }
  n_ = lower.rows();
  input_nonzeros_ = lower.nnz();
  parent_.assign(sz(n_), -1);
  column_count_.assign(sz(n_), 0);
  flag_.assign(sz(n_), -1);
  pattern_.assign(sz(n_), 0);
  filled_.assign(sz(n_), 0);
  work_.assign(sz(n_), 0.0);
  diagonal_.assign(sz(n_), 0.0);

  // Elimination tree and column counts in one pass. For each entry (i, k) above
  // the diagonal, walk from i up the tree marking the path; every node on it
  // gains a nonzero in column k of L.
  for (Int k = 0; k < n_; ++k) {
    flag_[sz(k)] = k;
    for (Int e = lower.row_begin(k); e < lower.row_end(k); ++e) {
      Int i = lower.index()[sz(e)];
      if (i >= k) continue;  // the strictly lower part of row k
      for (; flag_[sz(i)] != k; i = parent_[sz(i)]) {
        if (parent_[sz(i)] == -1) parent_[sz(i)] = k;
        ++column_count_[sz(i)];
        flag_[sz(i)] = k;
        if (parent_[sz(i)] == -1) break;
      }
    }
  }

  start_.assign(sz(n_) + 1, 0);
  for (Int k = 0; k < n_; ++k) start_[sz(k) + 1] = start_[sz(k)] + column_count_[sz(k)];
  index_.assign(sz(start_[sz(n_)]), 0);
  values_.assign(sz(start_[sz(n_)]), 0.0);
  return true;
}

bool LdlFactor::factorize(const SparseMatrix& lower, const LdlOptions& options,
                          std::string* error) {
  if (lower.rows() != n_ || start_.empty()) {
    if (error) *error = "factorize() called before analyse(), or with a different pattern";
    return false;
  }
  std::fill(filled_.begin(), filled_.end(), 0);
  std::fill(work_.begin(), work_.end(), 0.0);
  std::fill(flag_.begin(), flag_.end(), -1);
  positive_pivots_ = 0;

  for (Int k = 0; k < n_; ++k) {
    // Scatter row k of the lower triangle into the workspace, and collect the
    // pattern of row k of L by walking the elimination tree.
    Int top = n_;
    flag_[sz(k)] = k;
    work_[sz(k)] = 0.0;
    for (Int e = lower.row_begin(k); e < lower.row_end(k); ++e) {
      const Int i = lower.index()[sz(e)];
      if (i > k) continue;
      work_[sz(i)] += lower.value()[sz(e)];
      Int len = 0;
      for (Int node = i; flag_[sz(node)] != k; node = parent_[sz(node)]) {
        pattern_[sz(len++)] = node;
        flag_[sz(node)] = k;
        if (parent_[sz(node)] == -1) break;
      }
      while (len > 0) pattern_[sz(--top)] = pattern_[sz(--len)];
    }

    double pivot = work_[sz(k)];
    work_[sz(k)] = 0.0;
    for (; top < n_; ++top) {
      const Int i = pattern_[sz(top)];
      const double yi = work_[sz(i)];
      work_[sz(i)] = 0.0;
      for (Int p = start_[sz(i)]; p < start_[sz(i)] + filled_[sz(i)]; ++p)
        work_[sz(index_[sz(p)])] -= values_[sz(p)] * yi;
      const double l = yi / diagonal_[sz(i)];
      pivot -= l * yi;
      const Int p = start_[sz(i)] + filled_[sz(i)];
      index_[sz(p)] = k;
      values_[sz(p)] = l;
      ++filled_[sz(i)];
    }

    if (std::fabs(pivot) < options.minimum_pivot) {
      if (error) {
        *error = "pivot " + std::to_string(pivot) + " at column " +
                 std::to_string(k) +
                 " is too small; the matrix is not quasi-definite";
      }
      return false;
    }
    diagonal_[sz(k)] = pivot;
    if (pivot > 0.0) ++positive_pivots_;
  }
  return true;
}

void LdlFactor::solve(std::vector<double>* x) const {
  if (static_cast<Int>(x->size()) != n_) return;
  // L y = b
  for (Int k = 0; k < n_; ++k) {
    const double xk = (*x)[sz(k)];
    if (xk == 0.0) continue;
    for (Int p = start_[sz(k)]; p < start_[sz(k)] + filled_[sz(k)]; ++p)
      (*x)[sz(index_[sz(p)])] -= values_[sz(p)] * xk;
  }
  // D z = y
  for (Int k = 0; k < n_; ++k) (*x)[sz(k)] /= diagonal_[sz(k)];
  // L' x = z
  for (Int k = n_; k-- > 0;) {
    double sum = (*x)[sz(k)];
    for (Int p = start_[sz(k)]; p < start_[sz(k)] + filled_[sz(k)]; ++p)
      sum -= values_[sz(p)] * (*x)[sz(index_[sz(p)])];
    (*x)[sz(k)] = sum;
  }
}

double LdlFactor::fill_ratio() const {
  if (input_nonzeros_ <= 0) return 0.0;
  return static_cast<double>(values_.size() + sz(n_)) /
         static_cast<double>(input_nonzeros_);
}

}  // namespace sankhya
