#include "sankhya/lu.hpp"

#include <algorithm>
#include <cmath>

namespace sankhya {
namespace {

// The active submatrix during elimination, held both ways round. Markowitz needs
// row counts and column counts to pick a pivot, and elimination needs to walk a
// column to find the rows to update and a row to find the columns - so both
// orientations have to be maintained together.
struct Active {
  Int n = 0;
  // Column-wise storage. Entries are (row, value); removed entries are marked
  // with row = -1 rather than compacted, because compacting during elimination
  // costs more than the wasted scan.
  std::vector<std::vector<std::pair<Int, double>>> columns;
  // Row-wise storage of column indices only; values are read from `columns`.
  std::vector<std::vector<Int>> rows;
  std::vector<Int> row_count;
  std::vector<Int> column_count;
  std::vector<bool> row_done;
  std::vector<bool> column_done;

  double at(Int row, Int column) const {
    for (const auto& entry : columns[sz(column)]) {
      if (entry.first == row) return entry.second;
    }
    return 0.0;
  }
};

}  // namespace

bool LuFactor::factorize(const SparseMatrix& columns, const std::vector<Int>& basis,
                         const LuOptions& options, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error) *error = message;
    return false;
  };

  n_ = static_cast<Int>(basis.size());
  etas_.clear();
  u_rows_.assign(sz(n_), {});
  u_values_.assign(sz(n_), {});
  u_diagonal_.assign(sz(n_), 0.0);
  pivot_row_.assign(sz(n_), -1);
  pivot_column_.assign(sz(n_), -1);
  row_step_.assign(sz(n_), -1);
  column_step_.assign(sz(n_), -1);
  basis_nonzeros_ = 0;
  triangular_ = 0;

  Active active;
  active.n = n_;
  active.columns.assign(sz(n_), {});
  active.rows.assign(sz(n_), {});
  active.row_count.assign(sz(n_), 0);
  active.column_count.assign(sz(n_), 0);
  active.row_done.assign(sz(n_), false);
  active.column_done.assign(sz(n_), false);

  // `columns` is the matrix stored column-wise, so its rows are the original
  // matrix's columns.
  for (Int j = 0; j < n_; ++j) {
    const Int source = basis[sz(j)];
    if (source < 0 || source >= columns.rows()) return fail("basis index out of range");
    for (Int e = columns.row_begin(source); e < columns.row_end(source); ++e) {
      const Int row = columns.index()[sz(e)];
      const double value = columns.value()[sz(e)];
      if (std::fabs(value) <= options.drop_tolerance) continue;
      if (row >= n_) return fail("basis column has a row index outside the basis");
      active.columns[sz(j)].emplace_back(row, value);
      active.rows[sz(row)].push_back(j);
      active.row_count[sz(row)]++;
      active.column_count[sz(j)]++;
      basis_nonzeros_++;
    }
  }

  Int step = 0;

  // Records one pivot: the column becomes a column of U, and the multipliers
  // below the pivot become an eta of L.
  auto take_pivot = [&](Int row, Int column) {
    const double pivot = active.at(row, column);
    pivot_row_[sz(step)] = row;
    pivot_column_[sz(step)] = column;
    row_step_[sz(row)] = step;
    column_step_[sz(column)] = step;
    u_diagonal_[sz(step)] = pivot;

    Eta eta;
    eta.pivot_row = row;
    eta.pivot = pivot;
    for (const auto& entry : active.columns[sz(column)]) {
      if (entry.first < 0 || entry.first == row) continue;
      if (active.row_done[sz(entry.first)]) continue;
      eta.rows.push_back(entry.first);
      eta.values.push_back(entry.second);
    }
    etas_.push_back(std::move(eta));

    active.row_done[sz(row)] = true;
    active.column_done[sz(column)] = true;
    for (const Int c : active.rows[sz(row)]) {
      if (!active.column_done[sz(c)]) active.column_count[sz(c)]--;
    }
    for (const auto& entry : active.columns[sz(column)]) {
      if (entry.first >= 0 && !active.row_done[sz(entry.first)])
        active.row_count[sz(entry.first)]--;
    }
    ++step;
  };

  // Phase one: peel singletons. A column with one remaining entry, or a row with
  // one, needs no elimination at all - it is already triangular. Most of an LP
  // basis disappears here.
  bool progress = true;
  while (progress && step < n_) {
    progress = false;
    for (Int j = 0; j < n_ && step < n_; ++j) {
      if (active.column_done[sz(j)] || active.column_count[sz(j)] != 1) continue;
      Int row = -1;
      for (const auto& entry : active.columns[sz(j)]) {
        if (entry.first >= 0 && !active.row_done[sz(entry.first)]) row = entry.first;
      }
      if (row < 0) return fail("column " + std::to_string(j) + " became empty");
      take_pivot(row, j);
      progress = true;
    }
    for (Int i = 0; i < n_ && step < n_; ++i) {
      if (active.row_done[sz(i)] || active.row_count[sz(i)] != 1) continue;
      Int column = -1;
      for (const Int c : active.rows[sz(i)]) {
        if (!active.column_done[sz(c)]) column = c;
      }
      if (column < 0) return fail("row " + std::to_string(i) + " became empty");
      take_pivot(i, column);
      progress = true;
    }
  }
  triangular_ = step;

  // Phase two: Markowitz on what is left. The cost of a pivot is
  // (row_count - 1) * (column_count - 1), an estimate of the fill it creates,
  // and among cheap candidates only those large enough relative to their column
  // are acceptable.
  while (step < n_) {
    Int best_row = -1;
    Int best_column = -1;
    long long best_cost = -1;

    for (Int j = 0; j < n_; ++j) {
      if (active.column_done[sz(j)]) continue;
      double largest = 0.0;
      for (const auto& entry : active.columns[sz(j)]) {
        if (entry.first < 0 || active.row_done[sz(entry.first)]) continue;
        largest = std::fmax(largest, std::fabs(entry.second));
      }
      if (largest <= options.drop_tolerance) continue;

      for (const auto& entry : active.columns[sz(j)]) {
        const Int row = entry.first;
        if (row < 0 || active.row_done[sz(row)]) continue;
        if (std::fabs(entry.second) < options.stability_threshold * largest) continue;
        const long long cost =
            static_cast<long long>(active.row_count[sz(row)] - 1) *
            static_cast<long long>(active.column_count[sz(j)] - 1);
        if (best_cost < 0 || cost < best_cost) {
          best_cost = cost;
          best_row = row;
          best_column = j;
          if (cost == 0) break;  // a singleton is as good as it gets
        }
      }
      if (best_cost == 0) break;
    }

    if (best_row < 0) return fail("basis is singular: no acceptable pivot remains");

    const double pivot = active.at(best_row, best_column);
    take_pivot(best_row, best_column);

    // Eliminate: for every remaining row with an entry in the pivot column, add
    // a multiple of the pivot row.
    const Eta& eta = etas_.back();
    for (std::size_t k = 0; k < eta.rows.size(); ++k) {
      const Int target = eta.rows[k];
      const double factor = eta.values[k] / pivot;
      if (std::fabs(factor) <= options.drop_tolerance) continue;

      for (const Int column : active.rows[sz(best_row)]) {
        if (active.column_done[sz(column)]) continue;
        const double source = active.at(best_row, column);
        if (source == 0.0) continue;
        const double delta = -factor * source;

        bool found = false;
        for (auto& entry : active.columns[sz(column)]) {
          if (entry.first == target) {
            entry.second += delta;
            found = true;
            break;
          }
        }
        if (!found) {
          active.columns[sz(column)].emplace_back(target, delta);
          active.rows[sz(target)].push_back(column);
          active.row_count[sz(target)]++;
          active.column_count[sz(column)]++;
        }
      }
    }
  }

  // Collect U. Column `step` of U holds the entries of the pivot column that lie
  // in rows already eliminated, expressed in step coordinates.
  for (Int s = 0; s < n_; ++s) {
    const Int column = pivot_column_[sz(s)];
    for (const auto& entry : active.columns[sz(column)]) {
      const Int row = entry.first;
      if (row < 0) continue;
      const Int other = row_step_[sz(row)];
      if (other < 0 || other >= s) continue;  // strictly above the diagonal
      if (std::fabs(entry.second) <= options.drop_tolerance) continue;
      u_rows_[sz(s)].push_back(other);
      u_values_[sz(s)].push_back(entry.second);
    }
  }

  for (Int s = 0; s < n_; ++s) {
    if (std::fabs(u_diagonal_[sz(s)]) <= options.drop_tolerance)
      return fail("zero pivot at step " + std::to_string(s));
  }
  return true;
}

void LuFactor::ftran(std::vector<double>* x) const {
  // B permuted by the pivot order factors as L U, with L unit lower triangular.
  // Solving B x = b is a forward pass through L then a back substitution
  // through U.
  //
  // U is stored by column, so the back substitution has to be column oriented:
  // solve for one unknown, then push its contribution into the entries above
  // it. Writing it row-oriented instead - subtracting entries of column s from
  // z[s] - reads values that have not been computed yet, which is what the
  // first version of this did and why every non-triangular case came out wrong.
  std::vector<double>& v = *x;

  for (Int s = 0; s < static_cast<Int>(etas_.size()); ++s) {
    const Eta& eta = etas_[sz(s)];
    const double head = v[sz(eta.pivot_row)];
    if (head == 0.0) continue;
    for (std::size_t k = 0; k < eta.rows.size(); ++k) {
      v[sz(eta.rows[k])] -= (eta.values[k] / eta.pivot) * head;
    }
  }

  std::vector<double> z(sz(n_), 0.0);
  for (Int s = 0; s < n_; ++s) z[sz(s)] = v[sz(pivot_row_[sz(s)])];
  for (Int s = n_ - 1; s >= 0; --s) {
    const double value = z[sz(s)] / u_diagonal_[sz(s)];
    z[sz(s)] = value;
    if (value == 0.0) continue;
    for (std::size_t k = 0; k < u_rows_[sz(s)].size(); ++k) {
      z[sz(u_rows_[sz(s)][k])] -= u_values_[sz(s)][k] * value;
    }
  }
  for (Int s = 0; s < n_; ++s) v[sz(pivot_column_[sz(s)])] = z[sz(s)];
}

void LuFactor::btran(std::vector<double>* x) const {
  // B' = U' L'. U' is lower triangular in step coordinates and the column
  // storage of U is exactly the row storage of U', so that half is an ordinary
  // forward substitution. L' is unit upper triangular, so its solve is a
  // backward pass with no division - dividing by the pivot there, as the first
  // version did, scales the answer by the diagonal of U a second time.
  std::vector<double>& v = *x;

  std::vector<double> z(sz(n_), 0.0);
  for (Int s = 0; s < n_; ++s) z[sz(s)] = v[sz(pivot_column_[sz(s)])];
  for (Int s = 0; s < n_; ++s) {
    double value = z[sz(s)];
    for (std::size_t k = 0; k < u_rows_[sz(s)].size(); ++k) {
      value -= u_values_[sz(s)][k] * z[sz(u_rows_[sz(s)][k])];
    }
    z[sz(s)] = value / u_diagonal_[sz(s)];
  }

  std::vector<double> w(sz(n_), 0.0);
  for (Int s = 0; s < n_; ++s) w[sz(pivot_row_[sz(s)])] = z[sz(s)];

  for (Int s = static_cast<Int>(etas_.size()) - 1; s >= 0; --s) {
    const Eta& eta = etas_[sz(s)];
    double sum = 0.0;
    for (std::size_t k = 0; k < eta.rows.size(); ++k) {
      sum += (eta.values[k] / eta.pivot) * w[sz(eta.rows[k])];
    }
    w[sz(eta.pivot_row)] -= sum;
  }
  v = w;
}

Int LuFactor::nonzeros() const {
  Int total = n_;
  for (const Eta& eta : etas_) total += static_cast<Int>(eta.rows.size());
  for (const auto& column : u_rows_) total += static_cast<Int>(column.size());
  return total;
}

double LuFactor::fill_ratio() const {
  if (basis_nonzeros_ == 0) return 1.0;
  return static_cast<double>(nonzeros()) / static_cast<double>(basis_nonzeros_);
}

}  // namespace sankhya
