#include "sankhya/sparse.hpp"

#include <algorithm>
#include <cmath>

namespace sankhya {

SparseMatrix SparseMatrix::from_triplets(Int rows, Int cols,
                                         std::vector<Triplet> entries) {
  SparseMatrix a;
  a.rows_ = rows;
  a.cols_ = cols;

  // Counting sort by row, then sort each row by column. Cheaper than a general
  // sort over the whole triplet list and it keeps the ordering deterministic.
  std::vector<Int> count(sz(rows) + 1, 0);
  for (const Triplet& t : entries) count[sz(t.row) + 1]++;
  for (std::size_t i = 1; i < count.size(); ++i) count[i] += count[i - 1];

  std::vector<Triplet> ordered(entries.size());
  {
    std::vector<Int> cursor = count;
    for (const Triplet& t : entries) {
      ordered[sz(cursor[sz(t.row)]++)] = t;
    }
  }

  a.start_.assign(sz(rows) + 1, 0);
  a.index_.reserve(entries.size());
  a.value_.reserve(entries.size());

  for (Int r = 0; r < rows; ++r) {
    const std::size_t lo = sz(count[sz(r)]);
    const std::size_t hi = sz(count[sz(r) + 1]);
    std::sort(ordered.begin() + static_cast<std::ptrdiff_t>(lo),
              ordered.begin() + static_cast<std::ptrdiff_t>(hi),
              [](const Triplet& x, const Triplet& y) { return x.col < y.col; });

    for (std::size_t k = lo; k < hi;) {
      const Int col = ordered[k].col;
      double sum = 0.0;
      while (k < hi && ordered[k].col == col) {
        sum += ordered[k].value;
        ++k;
      }
      if (sum != 0.0) {
        a.index_.push_back(col);
        a.value_.push_back(sum);
      }
    }
    a.start_[sz(r) + 1] = static_cast<Int>(a.value_.size());
  }
  return a;
}

void SparseMatrix::multiply(const double* x, double* y) const {
  for (Int r = 0; r < rows_; ++r) {
    double sum = 0.0;
    const Int lo = start_[sz(r)];
    const Int hi = start_[sz(r) + 1];
    for (Int k = lo; k < hi; ++k) sum += value_[sz(k)] * x[index_[sz(k)]];
    y[r] = sum;
  }
}

void SparseMatrix::multiply_transpose(const double* x, double* y) const {
  for (Int c = 0; c < cols_; ++c) y[c] = 0.0;
  for (Int r = 0; r < rows_; ++r) {
    const double xr = x[r];
    if (xr == 0.0) continue;
    const Int lo = start_[sz(r)];
    const Int hi = start_[sz(r) + 1];
    for (Int k = lo; k < hi; ++k) y[index_[sz(k)]] += value_[sz(k)] * xr;
  }
}

SparseMatrix SparseMatrix::transpose() const {
  SparseMatrix t;
  t.rows_ = cols_;
  t.cols_ = rows_;
  t.start_.assign(sz(cols_) + 1, 0);
  t.index_.resize(value_.size());
  t.value_.resize(value_.size());

  for (const Int c : index_) t.start_[sz(c) + 1]++;
  for (std::size_t i = 1; i < t.start_.size(); ++i) t.start_[i] += t.start_[i - 1];

  std::vector<Int> cursor(t.start_.begin(), t.start_.end() - 1);
  for (Int r = 0; r < rows_; ++r) {
    const Int lo = start_[sz(r)];
    const Int hi = start_[sz(r) + 1];
    for (Int k = lo; k < hi; ++k) {
      const Int c = index_[sz(k)];
      const Int dest = cursor[sz(c)]++;
      t.index_[sz(dest)] = r;
      t.value_[sz(dest)] = value_[sz(k)];
    }
  }
  // Rows of A are visited in increasing order, so each column of the transpose
  // comes out already sorted by index.
  return t;
}

void SparseMatrix::row_norms(Norm norm, double* out) const {
  for (Int r = 0; r < rows_; ++r) {
    double acc = 0.0;
    const Int lo = start_[sz(r)];
    const Int hi = start_[sz(r) + 1];
    for (Int k = lo; k < hi; ++k) {
      const double v = std::fabs(value_[sz(k)]);
      acc = (norm == Norm::kInf) ? std::fmax(acc, v) : acc + v;
    }
    out[r] = acc;
  }
}

void SparseMatrix::col_norms(Norm norm, double* out) const {
  for (Int c = 0; c < cols_; ++c) out[c] = 0.0;
  for (std::size_t k = 0; k < value_.size(); ++k) {
    const double v = std::fabs(value_[k]);
    double& acc = out[index_[k]];
    acc = (norm == Norm::kInf) ? std::fmax(acc, v) : acc + v;
  }
}

void SparseMatrix::scale_rows(const double* d) {
  for (Int r = 0; r < rows_; ++r) {
    const double dr = d[r];
    const Int lo = start_[sz(r)];
    const Int hi = start_[sz(r) + 1];
    for (Int k = lo; k < hi; ++k) value_[sz(k)] *= dr;
  }
}

void SparseMatrix::scale_cols(const double* d) {
  for (std::size_t k = 0; k < value_.size(); ++k) value_[k] *= d[index_[k]];
}

bool SparseMatrix::validate(std::string* error) const {
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };
  if (rows_ < 0 || cols_ < 0) return fail("negative dimensions");
  if (start_.size() != sz(rows_) + 1) return fail("start has the wrong length");
  if (start_.front() != 0) return fail("start does not begin at 0");
  if (start_.back() != static_cast<Int>(value_.size()))
    return fail("start does not end at nnz");
  if (index_.size() != value_.size()) return fail("index and value length differ");

  for (Int r = 0; r < rows_; ++r) {
    const Int lo = start_[sz(r)];
    const Int hi = start_[sz(r) + 1];
    if (hi < lo) return fail("start is not monotone at row " + std::to_string(r));
    for (Int k = lo; k < hi; ++k) {
      const Int c = index_[sz(k)];
      if (c < 0 || c >= cols_)
        return fail("column index out of range at row " + std::to_string(r));
      if (k > lo && index_[sz(k) - 1] >= c)
        return fail("columns not strictly increasing in row " + std::to_string(r));
    }
  }
  return true;
}

}  // namespace sankhya
